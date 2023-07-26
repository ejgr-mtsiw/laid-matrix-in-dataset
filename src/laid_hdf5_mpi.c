/*
 ============================================================================
 Name        : laid_hdf5_mpi.c
 Author      : Eduardo Ribeiro
 Description : OpenMPI implementation of the LAID algorithm in C + HDF5
 ============================================================================
 */

#include "dataset.h"
#include "dataset_hdf5.h"
#include "dataset_hdf5_mpi.h"
#include "disjoint_matrix.h"
#include "disjoint_matrix_mpi.h"
#include "jnsq.h"
#include "set_cover.h"
#include "set_cover_hdf5_mpi.h"
#include "types/cover_t.h"
#include "types/dataset_hdf5_t.h"
#include "types/dataset_t.h"
#include "types/dm_t.h"
#include "types/steps_t.h"
#include "types/word_t.h"
#include "utils/bit.h"
#include "utils/block.h"
#include "utils/clargs.h"
#include "utils/math.h"
#include "utils/output.h"
#include "utils/ranks.h"
#include "utils/sort_r.h"
#include "utils/timing.h"

#include "hdf5.h"
#include "mpi.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

/**
 * If we have to build the disjoint matrices:
 * Each node root will open the original dataset file and read the contents to
 * memory. This allows us to save memory in each node, without sacrificing much
 * performance, because we're not having to send data across nodes.
 *
 * If the file already has a dataset with the name of the disjoing matrix
 * dataset we assume it was built in a previous itartion and skip ahead for the
 * cover algorithm.
 *
 * If we need to build the disjoint matrices: The node root(s) sort the dataset
 * in memory, remove duplicates and  adds jnsqs bits if necessary.
 *
 * Then they build a list of the steps needed to generate the disjoint matrix.
 * This list of steps allows us to create the disjoint matrix where the lines
 * represent observations and column are attributes. This simplifies the
 * building process for the matrix.
 *
 * Having access to the original dataset and the list of steps, each process
 * generates a part of the final matrix and stores it in the hdf5 file.
 *
 * We can't access easily the attribute data using the line dataset because they
 * are stored in words of size WORD_BITS (usually 64 bits). Meaning that we can
 * only read blocks of WORD_BITS attributes at once. This is wasteful and slow,
 * because of the file seeking time, and because we then need to extract the
 * info for one attribute from every word.
 *
 * To alleviate this we generate a new dataset where each line has the data for
 * an attribute and the columns represent the observations. We can now easily
 * get all the data for an attribute with a single read from the hdf5 file with
 * only one seek.
 *
 * TLDR:
 * Each node root
 *  - Reads dataset attributes from hdf5 file
 *  - Read dataset
 *  - Sort dataset
 *  - Remove duplicates
 *  - Add jnsqs
 *  - Builds steps for matrix generation
 *
 * All processes
 *  - Write disjoint matrix
 *  - Apply set covering algorithm
 *
 * Global root
 *  - Show solution
 */
int main(int argc, char** argv)
{
	/**
	 * Command line arguments set by the user
	 */
	clargs_t args;

	/**
	 * Parse command line arguments
	 */
	if (read_args(argc, argv, &args) == READ_CL_ARGS_NOK)
	{
		return EXIT_FAILURE;
	}

	/*
	 * Initialize MPI
	 */
	if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
	{
		printf("Error initializing MPI environment!\n");
		return EXIT_FAILURE;
	}

	/**
	 * rank of process
	 */
	int rank;

	/**
	 * number of processes
	 */
	int size;

	/**
	 * Global communicator group
	 */
	MPI_Comm comm = MPI_COMM_WORLD;

	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	MPI_Comm node_comm = MPI_COMM_NULL;

	/**
	 * Create node-local communicator
	 * This communicator is used to share memory with processes intranode
	 */
	MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
						&node_comm);

	int node_size, node_rank;

	MPI_Comm_size(node_comm, &node_size);
	MPI_Comm_rank(node_comm, &node_rank);

	/**
	 * Timing for the full operation
	 */
	time_t main_tick = 0, main_tock = 0;

	/**
	 * Local timing structures
	 */
	time_t tick = 0, tock = 0;

	if (rank == ROOT_RANK)
	{
		/**
		 * Timing for the full operation
		 */
		main_tick = time(0);
	}

	/**
	 * The dataset
	 */
	dataset_t dataset;
	init_dataset(&dataset);

	/**
	 * The HDF5 dataset file
	 */
	dataset_hdf5_t hdf5_dset;

	/**
	 * The disjoint matrix info
	 */
	dm_t dm;

	/**
	 * Open dataset file
	 */
	ROOT_SHOWS("Using dataset '%s'\n", args.filename);
	ROOT_SHOWS("Using %d processes\n\n", size);
	if (mpi_hdf5_open_dataset(args.filename, args.datasetname, comm,
							  MPI_INFO_NULL, &hdf5_dset)
		== NOK)
	{
		return EXIT_FAILURE;
	}

	/**
	 * We can jump straight to the set covering algorithm
	 * if we already have the matrix in the hdf5 dataset
	 */
	uint8_t skip_dm_creation
		= hdf5_dataset_exists(hdf5_dset.file_id, DM_LINE_DATA);

	if (skip_dm_creation)
	{
		/**
		 * We don't have to build the disjoint matrix!
		 */
		ROOT_SAYS("Disjoint matrix dataset found.\n\n");

		goto apply_set_cover;
	}

	/**
	 * We have to build the disjoint matrices.
	 */
	ROOT_SAYS("Disjoint matrix dataset not found.\n\n");
	ROOT_SAYS("Initializing MPI RMA: ");
	TICK;

	dataset.n_observations = hdf5_dset.dimensions[0];
	dataset.n_words		   = hdf5_dset.dimensions[1];

	/**
	 * Only rank 0 on a node actually allocates memory
	 */
	uint64_t dset_data_size = 0;
	if (node_rank == LOCAL_ROOT_RANK)
	{
		dset_data_size = dataset.n_observations * dataset.n_words;
	}

	char node_name[MPI_MAX_PROCESSOR_NAME];
	int node_str_len = 0;
	MPI_Get_processor_name(node_name, &node_str_len);

	word_t* dset_data		= NULL;
	MPI_Win win_shared_dset = MPI_WIN_NULL;
	MPI_Win_allocate_shared(dset_data_size * sizeof(word_t), sizeof(word_t),
							MPI_INFO_NULL, node_comm, &dset_data,
							&win_shared_dset);

	/**
	 * Set dataset data pointer
	 */
	if (node_rank == LOCAL_ROOT_RANK)
	{
		dataset.data = dset_data;
	}
	else
	{
		MPI_Aint win_size;
		int win_disp;
		MPI_Win_shared_query(win_shared_dset, 0, &win_size, &win_disp,
							 &dataset.data);
	}
	/**
	 *All table pointers should now point to copy on noderank 0
	 */
	TOCK;

	/**
	 * Setup dataset
	 */
	if (node_rank == LOCAL_ROOT_RANK)
	{
		ROOT_SAYS("Reading dataset: ");
		TICK;

		// Load dataset attributes
		hdf5_read_dataset_attributes(hdf5_dset.dataset_id, &dataset);

		// Load dataset data
		hdf5_read_dataset_data(hdf5_dset.dataset_id, dataset.data);

		TOCK;
		/**
		 * Print dataset details
		 */
		ROOT_SHOWS("  Classes = %d", dataset.n_classes);
		ROOT_SHOWS(" [%d bits]\n", dataset.n_bits_for_class);
		ROOT_SHOWS("  Attributes = %d \n", dataset.n_attributes);
		ROOT_SHOWS("  Observations = %d \n", dataset.n_observations);

		/**Sort dataset
		 *
		 */
		ROOT_SAYS("Sorting dataset: ");
		TICK;

		/**
		 * We need to know the number of longs in each line of the dataset
		 * so we can't use the standard qsort implementation
		 */
		sort_r(dataset.data, dataset.n_observations,
			   dataset.n_words * sizeof(word_t), compare_lines_extra,
			   &dataset.n_words);

		TOCK;

		/**
		 * Remove duplicates
		 */
		ROOT_SAYS("Removing duplicates: ");
		TICK;

		unsigned int duplicates = remove_duplicates(&dataset);

		TOCK;
		ROOT_SHOWS("  %d duplicate(s) removed\n", duplicates);

		/**
		 * Fill class arrays
		 */
		ROOT_SAYS("Checking classes: ");
		TICK;

		if (fill_class_arrays(&dataset) != OK)
		{
			return EXIT_FAILURE;
		}

		TOCK;

		for (unsigned int i = 0; i < dataset.n_classes; i++)
		{
			ROOT_SHOWS("  Class %d: ", i);
			ROOT_SHOWS("%d item(s)\n", dataset.n_observations_per_class[i]);
		}

		/**
		 * Set JNSQ
		 */
		ROOT_SAYS("Setting up JNSQ attributes: ");
		TICK;

		unsigned int max_jnsq = add_jnsqs(&dataset);

		TOCK;
		ROOT_SHOWS("  Max JNSQ: %d", max_jnsq);
		ROOT_SHOWS(" [%d bits]\n", dataset.n_bits_for_jnsqs);
	}

	/**
	 * End setup dataset
	 */

	dm.n_matrix_lines = 0;
	if (node_rank == LOCAL_ROOT_RANK)
	{
		dm.n_matrix_lines = get_dm_n_lines(&dataset);
	}

	steps_t* localsteps		 = NULL;
	MPI_Win win_shared_steps = MPI_WIN_NULL;
	MPI_Win_allocate_shared(dm.n_matrix_lines * sizeof(steps_t),
							sizeof(steps_t), MPI_INFO_NULL, node_comm,
							&localsteps, &win_shared_steps);

	/**
	 * The steps
	 */
	steps_t* steps = NULL;

	/**
	 * Set dataset data pointer
	 */
	if (node_rank == LOCAL_ROOT_RANK)
	{
		steps = localsteps;
	}
	else
	{
		MPI_Aint win_size;
		int win_disp;
		MPI_Win_shared_query(win_shared_steps, 0, &win_size, &win_disp, &steps);
	}
	/**
	 * All table pointers should now point to copy on noderank 0
	 */

	/**
	 * Setup steps
	 */
	if (node_rank == LOCAL_ROOT_RANK)
	{
		ROOT_SAYS("Generating matrix steps: ");
		TICK;

		uint32_t nc	   = dataset.n_classes;
		uint32_t no	   = dataset.n_observations;
		uint32_t* opc  = dataset.observations_per_class;
		uint32_t* nopc = dataset.n_observations_per_class;

		/**
		 * DO IT
		 */
		uint32_t cs = 0;

		for (uint32_t ca = 0; ca < nc - 1; ca++)
		{
			for (uint32_t ia = 0; ia < nopc[ca]; ia++)
			{
				for (uint32_t cb = ca + 1; cb < nc; cb++)
				{
					for (uint32_t ib = 0; ib < nopc[cb]; ib++)
					{
						steps[cs].indexA = opc[ca * no + ia];
						steps[cs].indexB = opc[cb * no + ib];

						cs++;
					}
				}
			}
		}

		free(dataset.n_observations_per_class);
		free(dataset.observations_per_class);

		dataset.n_observations_per_class = NULL;
		dataset.observations_per_class	 = NULL;

		TOCK;
		ROOT_SHOWS("  %d matrix steps generated\n", dm.n_matrix_lines);
	}

	uint64_t toshare[4];
	if (node_rank == 0)
	{
		toshare[0] = dataset.n_attributes;
		toshare[1] = dataset.n_observations;
		toshare[2] = dataset.n_words;
		toshare[3] = dm.n_matrix_lines;

		MPI_Bcast(&toshare, 4, MPI_UINT64_T, 0, node_comm);
	}
	else
	{
		MPI_Bcast(&toshare, 4, MPI_UINT64_T, 0, node_comm);

		dataset.n_attributes   = toshare[0];
		dataset.n_observations = toshare[1];
		dataset.n_words		   = toshare[2];
		dm.n_matrix_lines	   = toshare[3];
	}

	dm.steps	= steps;
	dm.s_offset = BLOCK_LOW(rank, size, dm.n_matrix_lines);
	dm.s_size	= BLOCK_SIZE(rank, size, dm.n_matrix_lines);

	ROOT_SAYS("Building disjoint matrix:\n");

	if (rank == ROOT_RANK)
	{
		double matrix_size
			= ((double) dm.n_matrix_lines
			   * (dataset.n_attributes + dataset.n_bits_for_jnsqs))
			/ (1024 * 1024 * 8);
		ROOT_SHOWS("  Estimated disjoint matrix size: %3.2fMB\n", matrix_size);
	}

	ROOT_SAYS("  Line dataset done: ");
	TICK;

	// MPI_Barrier(node_comm);

	/**
	 *Build part of the disjoint matrix and store it in the hdf5 file
	 */
	mpi_create_line_dataset(&hdf5_dset, &dataset, &dm);

	TOCK;
	ROOT_SAYS("  Column dataset done: ");
	TICK;

	// MPI_Barrier(comm);

	mpi_create_column_dataset(&hdf5_dset, &dataset, &dm, rank, size);

	MPI_Barrier(comm);

	TOCK;

	/**
	 * We no longer need the original dataset
	 * or the steps to generate the disjoint matrix
	 */
	MPI_Win_free(&win_shared_dset);
	MPI_Win_free(&win_shared_steps);

	dataset.data = NULL;
	dm.steps	 = NULL;

	free_dataset(&dataset);
	free_dm(&dm);

apply_set_cover:
	/**
	 * All:
	 *  - Setup line covered array -> 0
	 *  - Setup attributes totals -> 0
	 *
	 * ROOT:
	 *  - Reads the global attributes totals
	 *loop:
	 *  - Selects the best one and blacklists it
	 *  - Sends attribute id to everyone else
	 *
	 * ~ROOT:
	 *  - Wait for attribute message
	 *
	 * ALL:
	 *  - Black list their lines covered by this attribute
	 *  - Update atributes totals
	 *  - MPI_Reduce attributes totals
	 *
	 * ROOT:
	 *  - Subtract atribute totals from global attributes total
	 *  - if there are still lines to blacklist:
	 *   - Goto loop
	 *  - else:
	 *   - Show solution
	 */

	/**
	 * We no longer need to keep the original dataset open
	 */
	H5Dclose(hdf5_dset.dataset_id);

	ROOT_SAYS("Applying set covering algorithm:\n");
	TICK;

	cover_t cover;
	init_cover(&cover);

	/**
	 * Open the line dataset
	 */
	hid_t d_id = H5Dopen2(hdf5_dset.file_id, DM_LINE_DATA, H5P_DEFAULT);
	assert(d_id != NOK);

	dataset_hdf5_t line_dset_id;
	line_dset_id.file_id	= hdf5_dset.file_id;
	line_dset_id.dataset_id = d_id;
	hdf5_get_dataset_dimensions(d_id, line_dset_id.dimensions);

	/**
	 * Open the column dataset
	 */
	d_id = H5Dopen2(hdf5_dset.file_id, DM_COLUMN_DATA, H5P_DEFAULT);
	assert(d_id != NOK);

	dataset_hdf5_t column_dset_id;
	column_dset_id.file_id	  = hdf5_dset.file_id;
	column_dset_id.dataset_id = d_id;
	hdf5_get_dataset_dimensions(d_id, column_dset_id.dimensions);

	/**
	 * If we skipped the matriz generation, dataset and dm are empty.
	 * So we need to read the attributes from the dataset
	 */
	hdf5_read_attribute(line_dset_id.dataset_id, N_MATRIX_LINES_ATTR,
						H5T_NATIVE_UINT32_g, &cover.n_matrix_lines);
	hdf5_read_attribute(line_dset_id.dataset_id, N_ATTRIBUTES_ATTR,
						H5T_NATIVE_UINT32_g, &cover.n_attributes);

	cover.n_words_in_a_line = line_dset_id.dimensions[1];

	/**
	 * Each process only needs acces to some rows,
	 * we don't need the entire column (attribute data)
	 */
	uint32_t n_words_in_a_column = cover.n_matrix_lines / WORD_BITS
		+ (cover.n_matrix_lines % WORD_BITS != 0);

	cover.column_offset_words = BLOCK_LOW(rank, size, n_words_in_a_column);
	cover.column_n_words	  = BLOCK_SIZE(rank, size, n_words_in_a_column);

	cover.covered_lines
		= (word_t*) calloc(cover.column_n_words, sizeof(word_t));
	cover.attribute_totals
		= (uint32_t*) calloc(cover.n_attributes, sizeof(uint32_t));

	/**
	 * Global totals. Only root needs these
	 */
	uint32_t* global_attribute_totals = NULL;
	uint32_t* attribute_totals_buffer = NULL;

	/**
	 * Number of uncovered lines. Only root uses this.
	 */
	uint32_t n_uncovered_lines = 0;

	if (rank == ROOT_RANK)
	{
		global_attribute_totals
			= (uint32_t*) calloc(cover.n_attributes, sizeof(uint32_t));

		attribute_totals_buffer
			= (uint32_t*) calloc(cover.n_attributes, sizeof(uint32_t));

		cover.selected_attributes
			= (word_t*) calloc(cover.n_words_in_a_line, sizeof(word_t));

		read_initial_attribute_totals(hdf5_dset.file_id,
									  global_attribute_totals);

		/**
		 * No line is covered so far
		 */
		n_uncovered_lines = cover.n_matrix_lines;
	}

	while (true)
	{
		int64_t best_attribute = 0;

		if (rank == ROOT_RANK)
		{
			best_attribute = get_best_attribute_index(global_attribute_totals,
													  cover.n_attributes);

			ROOT_SHOWS("  Selected attribute #%ld, ", best_attribute);
			ROOT_SHOWS("covers %d lines ",
					   global_attribute_totals[best_attribute]);
			TOCK;
			TICK;

			mark_attribute_as_selected(&cover, best_attribute);

			/**
			 * Update number of lines remaining
			 */
			n_uncovered_lines -= global_attribute_totals[best_attribute];

			/**
			 * If we covered all of them, we can leav earlier
			 */
			if (n_uncovered_lines == 0)
			{
				best_attribute = -1;
			}
		}

		MPI_Bcast(&best_attribute, 1, MPI_INT64_T, 0, comm);

		if (best_attribute < 0)
		{
			goto show_solution;
		}

		/**
		 * Even if this process has no lines it needs to participate in
		 * the MPI_Reduce
		 */
		if (cover.column_n_words == 0)
		{
			SAY("NOTHING TO DO!\n");
			goto mpi_reduce;
		}

		word_t* column = (word_t*) calloc(cover.column_n_words, sizeof(word_t));

		get_column(column_dset_id.dataset_id, best_attribute,
				   cover.column_offset_words, cover.column_n_words, column);

		update_attribute_totals_mpi(&cover, &line_dset_id, column);

		update_covered_lines_mpi(column, cover.column_n_words,
								 cover.covered_lines);

mpi_reduce:
		MPI_Reduce(cover.attribute_totals, attribute_totals_buffer,
				   cover.n_attributes, MPI_INT, MPI_SUM, 0, comm);

		/**
		 * Update global totals
		 */
		if (rank == ROOT_RANK)
		{
			for (uint32_t a = 0; a < cover.n_attributes; a++)
			{
				global_attribute_totals[a] -= attribute_totals_buffer[a];
			}
		}

		/**
		 * Reset local totals
		 */
		memset(cover.attribute_totals, 0,
			   cover.n_attributes * sizeof(uint32_t));
	}

show_solution:
	// wait for everyone
	MPI_Barrier(comm);

	if (rank == ROOT_RANK)
	{
		print_solution(stdout, &cover);
		fprintf(stdout, "All done! ");

		main_tock = time(0);
		fprintf(stdout, "[%lds]\n", main_tock - main_tick);

		free(global_attribute_totals);
		global_attribute_totals = NULL;

		free(attribute_totals_buffer);
		attribute_totals_buffer = NULL;

		free_cover(&cover);
	}

	// Close dataset files
	H5Dclose(line_dset_id.dataset_id);
	H5Dclose(column_dset_id.dataset_id);
	H5Fclose(hdf5_dset.file_id);

	/* shut down MPI */
	MPI_Finalize();

	return EXIT_SUCCESS;
}
