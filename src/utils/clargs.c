/*
 ============================================================================
 Name        : utils/clargs.c
 Author      : Eduardo Ribeiro
 Description : Structures and functions to manage command line arguments
 ============================================================================
 */

#include "utils/clargs.h"

#include "utils/cargs.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int read_args(int argc, char** argv, clargs_t* args)
{
	char identifier;
	const char* value;
	cag_option_context context;

	args->datasetname = NULL;
	args->filename	  = NULL;

	/**
	 * This is the main configuration of all options available.
	 */
	cag_option options[] = { { .identifier	   = 'f',
							   .access_letters = "f",
							   .access_name	   = NULL,
							   .value_name	   = "filename",
							   .description	   = "HDF5 dataset filename" },

							 { .identifier	   = 'd',
							   .access_letters = "d",
							   .access_name	   = NULL,
							   .value_name	   = "dataset",
							   .description	   = "Dataset identifier" },

							 { .identifier	   = 'h',
							   .access_letters = "h",
							   .access_name	   = "help",
							   .description	   = "Shows the command help"

							 } };

	/**
	 * Now we just prepare the context and iterate over all options. Simple!
	 */
	cag_option_prepare(&context, options, CAG_ARRAY_SIZE(options), argc, argv);
	while (cag_option_fetch(&context))
	{
		identifier = cag_option_get(&context);
		switch (identifier)
		{
			case 'f':
				value		   = cag_option_get_value(&context);
				args->filename = value;
				break;
			case 'd':
				value			  = cag_option_get_value(&context);
				args->datasetname = value;
				break;
			case 'h':
				printf("Usage: %s [OPTION]...\n", argv[0]);
				cag_option_print(options, CAG_ARRAY_SIZE(options), stdout);
				return READ_CL_ARGS_NOK;
		}
	}

	if (args->filename == NULL || args->datasetname == NULL)
	{
		printf("Usage: %s [OPTION]...\n", argv[0]);
		cag_option_print(options, CAG_ARRAY_SIZE(options), stdout);
		return READ_CL_ARGS_NOK;
	}

	return READ_CL_ARGS_OK;
}
