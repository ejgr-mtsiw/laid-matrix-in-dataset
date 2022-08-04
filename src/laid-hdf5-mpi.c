/*
 ============================================================================
 Name        : laid-hdf5-mpi.c
 Author      : Eduardo Ribeiro
 Description : OpenMPI implementation of the LAID algorithm in C + HDF5
 ============================================================================
 */

#include "dataset.h"
#include "disjoint_matrix.h"
#include "hdf5_dataset.h"
#include "jnsq.h"
#include "mpi_disjoint_matrix.h"
#include "mpi_hdf5_dataset.h"
#include "sort_r.h"
#include "types/dataset_t.h"
#include "types/dm_t.h"
#include "types/hdf5_dataset_t.h"
#include "types/steps_t.h"
#include "types/word_t.h"
#include "utils/block.h"
#include "utils/timing.h"

#include "mpi.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <utils/clargs.h>

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
	hdf5_dataset_t hdf5_dset;
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
		if (rank == 0)
		{
			TICK;

			fprintf(stdout, "- Loading dataset data\n - ");
		}
		/**
		 * The dataset
		 */
		dataset_t dataset;
		init_dataset(&dataset);
		dataset.data = calloc(n_observations * n_words, sizeof(word_t));

		// Load dataset attributes
		hdf5_read_dataset_attributes(hdf5_dset.dataset_id, &dataset);
		// Load dataset data
		hdf5_read_data(hdf5_dset.dataset_id, &dataset);

		if (rank == 0)
		{
			print_dataset_details(stdout, &dataset);

			fprintf(stdout, " - Finished loading dataset data ");

			TOCK(stdout)
			TICK;

			// Sort dataset
			fprintf(stdout, "- Sorting dataset\n");
		}

		// We need to know the number of longs in each line of the dataset
		// so we can't use the standard qsort implementation
		sort_r(dataset.data, dataset.n_observations,
			   dataset.n_words * sizeof(word_t), compare_lines_extra,
			   &dataset.n_words);

		if (rank == 0)
		{
			fprintf(stdout, " - Sorted dataset");
			TOCK(stdout)
			TICK;

			// Remove duplicates
			fprintf(stdout, "- Removing duplicates:\n");
		}

		unsigned int duplicates = remove_duplicates(&dataset);

		if (rank == 0)
		{
			fprintf(stdout, " - %d duplicate(s) removed ", duplicates);
			TOCK(stdout)
			TICK;

			// Fill class arrays
			fprintf(stdout, "- Checking classes: ");
		}

		if (fill_class_arrays(&dataset) != OK)
		{
			return EXIT_FAILURE;
		}

		if (rank == 0)
		{
			TOCK(stdout)

			for (unsigned int i = 0; i < dataset.n_classes; i++)
			{
				fprintf(stdout, " - class %d: %d item(s)\n", i,
						dataset.n_observations_per_class[i]);
			}

			TICK;

			// Set JNSQ
			fprintf(stdout, "- Setting up JNSQ attributes:\n");
		}

		unsigned int max_jnsq = add_jnsqs(&dataset);

		if (rank == 0)
		{
			fprintf(stdout, " - Max JNSQ: %d [%d bits] ", max_jnsq,
					dataset.n_bits_for_jnsqs);
			TOCK(stdout)
		}

		uint32_t n_matrix_lines = get_dm_n_lines(&dataset);
		//uint32_t start			= BLOCK_LOW(rank, size, n_matrix_lines);
		//uint32_t n_lines		= BLOCK_SIZE(rank, size, n_matrix_lines);

		//printf("[%d] s:%d %d/%d\n", rank, start, n_lines, n_matrix_lines);

		/**
		 * The steps
		 */
		dm_t dm;

		dm.n_matrix_lines = n_matrix_lines;
		dm.steps		  = (steps_t*) malloc(n_matrix_lines * sizeof(steps_t));

		steps_t* steps = dm.steps;

		if (rank == 0)
		{
			TICK;

			fprintf(stdout, "- Generating matrix steps\n");
		}

		uint32_t nc	   = dataset.n_classes;
		uint32_t no	   = dataset.n_observations;
		uint32_t* opc  = dataset.observations_per_class;
		uint32_t* nopc = dataset.n_observations_per_class;

		// DO IT
		// Current step (global)
		//uint32_t gcs = 0;

		// My current step
		uint32_t cs = 0;

		for (uint32_t ca = 0; ca < nc - 1; ca++)
		{
			for (uint32_t ia = 0; ia < nopc[ca]; ia++)
			{
				for (uint32_t cb = ca + 1; cb < nc; cb++)
				{
					for (uint32_t ib = 0; ib < nopc[cb]; ib++/*, gcs++*/)
					{
//						if (gcs < start)
//						{
//							continue;
//						}

//						if (cs >= n_lines)
//						{
//							goto steps_done;
//						}

						steps[cs].indexA = opc[ca * no + ia];
						steps[cs].indexB = opc[cb * no + ib];

						cs++;
					}
				}
			}
		}

//steps_done:

//		for (uint32_t i = start; i < start+n_lines; i++)
//		{
//			printf("[%d] [%d]: %d ^ %d\n", rank, i, steps[i].indexA + 1,
//				   steps[i].indexB + 1);
//		}

		if (rank == 0)
		{
			fprintf(stdout, " - Finished generating matrix steps ");
			TOCK(stdout)
		}

		if (rank == 0)
		{
			fprintf(stdout, "- Building disjoint matrix\n");

			TICK;
		}

		// Build part of the disjoint matrix and store it in the hdf5 file
		mpi_create_line_dataset(&hdf5_dset, &dataset, &dm, rank, size);

		MPI_Barrier(comm);


		if (rank == 0)
		{
			fprintf(stdout, " - Finished building disjoint matrix [1/2] ");
			TOCK(stdout)
			TICK;
		}

		mpi_create_column_dataset(&hdf5_dset, &dataset, &dm, rank, size);

		MPI_Barrier(comm);
		if (rank == 0)
		{
			fprintf(stdout, " - Finished building disjoint matrix [2/2] ");
			TOCK(stdout)
		}

		dataset.data = NULL;
		free_dataset(&dataset);
		hdf5_close_dataset(&hdf5_dset);

		goto the_end;
	}

	//		unsigned long matrix_lines = get_dm_n_lines(&dataset);
	//
	//		double matrixsize
	//			= (matrix_lines * (dataset.n_attributes +
	// dataset.n_bits_for_class)) 			/ (1024 * 1024 * 8);
	//
	//		fprintf(stdout, "\nBuilding disjoint matrix.\n");
	//		fprintf(stdout,
	//				"Estimated disjoint matrix size: %lu lines
	//[%0.2fMB]\n", 				matrix_lines, matrixsize);

	// Sync everyone before starting building the matrix dataset
	// MPI_Barrier(comm);

	// Build part of the disjoint matrix and store it in the hdf5 file
	//		mpi_create_disjoint_matrix(args.filename, &dataset, rank, size,
	// comm, 								   MPI_INFO_NULL);

	//		fprintf(stdout, "Finished building disjoint matrix ");
	//		TOCK(stdout)

	/**
	 * From this point forward we no longer need the dataset
	 */
	// free_dataset(&dataset);

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
