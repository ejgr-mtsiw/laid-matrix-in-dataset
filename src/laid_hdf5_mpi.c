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
#include "types/dataset_hdf5_t.h"
#include "types/dataset_t.h"
#include "types/dm_t.h"
#include "types/steps_t.h"
#include "types/word_t.h"
#include "utils/block.h"
#include "utils/clargs.h"
#include "utils/math.h"
#include "utils/sort_r.h"
#include "utils/timing.h"

#include "mpi.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

/**
 * Reads dataset attributes from hdf5 file
 * Read dataset
 * Sort dataset
 * Remove duplicates
 * Add jnsqs
 * Write disjoint matrix
 * Apply set covering algorithm
 * Show solution
 */
int main(int argc, char** argv)
{

	/**
	 * Command line arguments set by the user
	 */
	clargs_t args;

	// Parse command line arguments
	if (read_args(argc, argv, &args) == READ_CL_ARGS_NOK)
	{
		return EXIT_FAILURE;
	}

	SETUP_TIMING

	/*
	 * Initialize MPI
	 */
	if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
	{
		printf("Error initializing MPI environment!\n");
		return EXIT_FAILURE;
	}

	// rank of process
	int rank;
	// number of processes
	int size;

	// Global communicator group
	MPI_Comm comm = MPI_COMM_WORLD;

	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	MPI_Comm node_comm = MPI_COMM_NULL;

	// Create node-local communicator
	MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
						&node_comm);

	int node_size, node_rank;

	MPI_Comm_size(node_comm, &node_size);
	MPI_Comm_rank(node_comm, &node_rank);

	// Open dataset
	dataset_hdf5_t hdf5_dset;
	mpi_hdf5_open_dataset(args.filename, args.datasetname, comm, MPI_INFO_NULL,
						  &hdf5_dset);

	uint32_t n_observations = hdf5_dset.dimensions[0];
	uint32_t n_words		= hdf5_dset.dimensions[1];

	struct timespec main_tick, main_tock;
	if (rank == 0)
	{
		// Timing for the full operation
		clock_gettime(CLOCK_MONOTONIC_RAW, &main_tick);
	}

	// We can jump straight to the set covering algorithm
	// if we already have the matrix in the hdf5 dataset
	// Only root checks to avoid having all processes trying
	// to open and read the file at the same time
	uint8_t skip_dm_creation = 0;
	if (rank == 0)
	{
		skip_dm_creation = hdf5_dataset_exists(hdf5_dset.file_id, DM_LINE_DATA);

		printf("- Disjoint matrix dataset %sfound!.\n",
			   skip_dm_creation ? "" : "not ");
	}

	MPI_Bcast(&skip_dm_creation, 1, MPI_INT8_T, 0, comm);

	if (!skip_dm_creation)
	{
		TICK;

		// Only rank 0 on a node actually allocates memory
		uint64_t localtablesize = 0;
		if (node_rank == 0)
		{
			localtablesize = n_observations * n_words;
		}

		char node_name[MPI_MAX_PROCESSOR_NAME];
		int node_str_len = 0;
		MPI_Get_processor_name(node_name, &node_str_len);

		// debug info
		//		printf(
		//			"Rank %d of %d, rank %d of %d in node <%s>, localtablesize
		//%lu\n", 			rank, size, node_rank, node_size, node_name,
		// localtablesize);

		word_t* localtable		= NULL;
		MPI_Win win_shared_dset = MPI_WIN_NULL;
		MPI_Win_allocate_shared(localtablesize * sizeof(word_t), sizeof(word_t),
								MPI_INFO_NULL, node_comm, &localtable,
								&win_shared_dset);

		/**
		 * The dataset
		 */
		dataset_t dataset;

		init_dataset(&dataset);

		// Set dataset data pointer
		if (node_rank == 0)
		{
			dataset.data = localtable;
		}
		else
		{
			MPI_Aint win_size;
			int win_disp;
			MPI_Win_shared_query(win_shared_dset, 0, &win_size, &win_disp,
								 &dataset.data);
		}

		if (rank == 0)
		{
			fprintf(stdout, "- Finished MPI RMA Init ");
			TOCK(stdout)
		}

		// All table pointers should now point to copy on noderank 0
		// Setup dataset
		//MPI_Win_fence(0, win_shared_dset);
		MPI_Barrier(node_comm);

		if (node_rank == 0)
		{
			TICK;

			fprintf(stdout, "- Loading dataset data\n - ");

			// Load dataset attributes
			hdf5_read_dataset_attributes(hdf5_dset.dataset_id, &dataset);
			// Load dataset data
			hdf5_read_data(hdf5_dset.dataset_id, &dataset);

			print_dataset_details(stdout, &dataset);

			fprintf(stdout, " - Finished loading dataset data ");

			TOCK(stdout)
			TICK;

			// Sort dataset
			fprintf(stdout, "- Sorting dataset\n");

			// We need to know the number of longs in each line of the dataset
			// so we can't use the standard qsort implementation
			sort_r(dataset.data, dataset.n_observations,
				   dataset.n_words * sizeof(word_t), compare_lines_extra,
				   &dataset.n_words);

			fprintf(stdout, " - Sorted dataset");
			TOCK(stdout)
			TICK;

			// Remove duplicates
			fprintf(stdout, "- Removing duplicates:\n");

			unsigned int duplicates = remove_duplicates(&dataset);

			fprintf(stdout, " - %d duplicate(s) removed ", duplicates);
			TOCK(stdout)
			TICK;

			// Fill class arrays
			fprintf(stdout, "- Checking classes: ");

			if (fill_class_arrays(&dataset) != OK)
			{
				return EXIT_FAILURE;
			}

			TOCK(stdout)

			for (unsigned int i = 0; i < dataset.n_classes; i++)
			{
				fprintf(stdout, " - class %d: %d item(s)\n", i,
						dataset.n_observations_per_class[i]);
			}

			TICK;

			// Set JNSQ
			fprintf(stdout, "- Setting up JNSQ attributes:\n");

			unsigned int max_jnsq = add_jnsqs(&dataset);

			fprintf(stdout, " - Max JNSQ: %d [%d bits] ", max_jnsq,
					dataset.n_bits_for_jnsqs);
			TOCK(stdout)
		}

		// End setup dataset
		//MPI_Win_fence(0, win_shared_dset);
		MPI_Barrier(node_comm);

		// Only rank 0 on a node actually allocates memory
		uint64_t n_matrix_lines = 0;
		if (node_rank == 0)
		{
			// Round up so we don't have so much trouble with
			// columns dataset
			n_matrix_lines = roundUp(get_dm_n_lines(&dataset), WORD_BITS);
		}

		// debug info
		//		printf("Rank %d of %d, rank %d of %d in node <%s>, local_dm_size
		//%lu\n", 			   rank, size, node_rank, node_size, node_name,
		// n_matrix_lines);

		steps_t* localsteps		 = NULL;
		MPI_Win win_shared_steps = MPI_WIN_NULL;
		MPI_Win_allocate_shared(n_matrix_lines * sizeof(steps_t),
								sizeof(steps_t), MPI_INFO_NULL, node_comm,
								&localsteps, &win_shared_steps);

		/**
		 * The steps
		 */
		steps_t* steps = NULL;

		// Set dataset data pointer
		if (node_rank == 0)
		{
			steps = localsteps;
		}
		else
		{
			MPI_Aint win_size;
			int win_disp;
			MPI_Win_shared_query(win_shared_steps, 0, &win_size, &win_disp,
								 &steps);
		}

		// All table pointers should now point to copy on noderank 0
		// Setup steps
		//MPI_Win_fence(0, win_shared_steps);
		MPI_Barrier(node_comm);

		if (node_rank == 0)
		{
			TICK;

			fprintf(stdout, "- Generating matrix steps\n");

			uint32_t nc	   = dataset.n_classes;
			uint32_t no	   = dataset.n_observations;
			uint32_t* opc  = dataset.observations_per_class;
			uint32_t* nopc = dataset.n_observations_per_class;

			// DO IT
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

			//			for (uint64_t i=0;i<n_matrix_lines;i++){
			//				printf("[%lu]: %d ^ %d\n", i, steps[i].indexA+1,
			// steps[i].indexB+1);
			//			}

			fprintf(stdout, " - Finished generating matrix steps ");
			TOCK(stdout)
		}

		// end setup steps
		//MPI_Win_fence(0, win_shared_steps);
		MPI_Barrier(comm);

		if (rank == 0)
		{
			fprintf(stdout, "- Broadcasting attributes\n");
		}

		dm_t dm;
		dm.steps		  = steps;
		dm.n_matrix_lines = n_matrix_lines;

		uint64_t toshare[4];
		if (rank == 0)
		{
			toshare[0] = dataset.n_attributes;
			toshare[1] = dataset.n_observations;
			toshare[2] = dataset.n_words;
			toshare[3] = n_matrix_lines;
		}
		MPI_Bcast(&toshare, 4, MPI_UINT64_T, 0, comm);

		if (rank == 0)
		{
			dm.s_offset = 0;
			dm.s_size	= BLOCK_SIZE(rank, size, n_matrix_lines);
		}
		else
		{
			dataset.n_attributes   = toshare[0];
			dataset.n_observations = toshare[1];
			dataset.n_words		   = toshare[2];
			dm.n_matrix_lines	   = toshare[3];
			dm.s_offset			   = BLOCK_LOW(rank, size, dm.n_matrix_lines);
			dm.s_size			   = BLOCK_SIZE(rank, size, dm.n_matrix_lines);
		}

//		for (int r = 0; r < size; r++)
//		{
//			if (r == rank)
//			{
//				printf("[%d] o:%d, s:%d, n:%d\n", rank, dm.s_offset, dm.s_size,
//					   dm.n_matrix_lines);
//			}
//			sleep(1);
//		}

		if (rank == 0)
		{
			fprintf(stdout, " - Finished broadcasting attributes\n");
			fprintf(stdout, "- Building disjoint matrix\n");
			TICK;
		}

		MPI_Barrier(comm);
		// Build part of the disjoint matrix and store it in the hdf5 file
		mpi_create_line_dataset(&hdf5_dset, &dataset, &dm);

		MPI_Barrier(comm);
		if (rank == 0)
		{
			fprintf(stdout, " - Finished building disjoint matrix [1/2] ");
			TOCK(stdout)
			TICK;
		}

		MPI_Barrier(comm);

		mpi_create_column_dataset(&hdf5_dset, &dataset, &dm, rank, size);

		MPI_Barrier(comm);
		if (rank == 0)
		{
			fprintf(stdout, " - Finished building disjoint matrix [2/2] ");
			TOCK(stdout)
		}

		MPI_Win_free(&win_shared_dset);
		MPI_Win_free(&win_shared_steps);

		dataset.data = NULL;
		free_dataset(&dataset);
		hdf5_close_dataset(&hdf5_dset);
	}
	//	//
	//	//	cover_t cover;
	//	//	init_cover(&cover);
	//	//
	//	//	TICK
	//	//	fprintf(stdout, "\nApplying set covering algorithm.\n");
	//	//	if (calculate_solution(args.filename, &cover) != OK) {
	//	//		return EXIT_FAILURE;
	//	//	}
	//	//
	//	//	print_solution(stdout, &cover);
	//	//
	//	//	free_cover(&cover);
	//	//	TOCK(stdout)
	goto the_end;
the_end:
	//  Wait for everyone
	MPI_Barrier(comm);

	if (rank == 0)
	{
		fprintf(stdout, "All done! ");

		clock_gettime(CLOCK_MONOTONIC_RAW, &main_tock);
		fprintf(stdout, "[%0.3fs]\n",
				(main_tock.tv_nsec - main_tick.tv_nsec) / 1000000000.0
					+ (main_tock.tv_sec - main_tick.tv_sec));
	}

	/* shut down MPI */
	MPI_Finalize();

	return EXIT_SUCCESS;
}
