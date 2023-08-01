/*
 ============================================================================
 Name        : dataset_hdf5.c
 Author      : Eduardo Ribeiro
 Description : Structures and functions to manage HDF5 datasets
 ============================================================================
 */

#include "dataset_hdf5.h"

#include "types/dataset_t.h"
#include "types/oknok_t.h"
#include "types/word_t.h"

#include "hdf5.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

bool hdf5_dataset_exists(const hid_t file_id, const char* datasetname)
{
	return (H5Lexists(file_id, datasetname, H5P_DEFAULT) > 0);
}

bool hdf5_file_has_dataset(const char* filename, const char* datasetname)
{
	// Open the data file
	hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file_id < 1)
	{
		// Error creating file
		fprintf(stderr, "Error opening file %s\n", filename);
		return false;
	}

	bool exists = hdf5_dataset_exists(file_id, datasetname);

	H5Fclose(file_id);

	return exists;
}

oknok_t hdf5_read_dataset_attributes(hid_t dataset_id, dataset_t* dataset)
{
	uint32_t n_classes = 0;
	hdf5_read_attribute(dataset_id, N_CLASSES_ATTR, H5T_NATIVE_UINT32,
						&n_classes);

	if (n_classes < 2)
	{
		fprintf(stderr, "Dataset must have at least 2 classes\n");
		return NOK;
	}

	// Number of observations (lines) in the dataset
	uint32_t n_observations = 0;
	hdf5_read_attribute(dataset_id, N_OBSERVATIONS_ATTR, H5T_NATIVE_UINT32,
						&n_observations);

	if (n_observations < 2)
	{
		fprintf(stderr, "Dataset must have at least 2 observations\n");
		return NOK;
	}

	// Number of attributes in the dataset
	uint32_t n_attributes = 0;
	hdf5_read_attribute(dataset_id, N_ATTRIBUTES_ATTR, H5T_NATIVE_UINT32,
						&n_attributes);

	if (n_attributes < 1)
	{
		fprintf(stderr, "Dataset must have at least 1 attribute\n");
		return NOK;
	}

	// Store data
	dataset->n_attributes	  = n_attributes;
	dataset->n_bits_for_class = (uint8_t) ceil(log2(n_classes));
	dataset->n_bits_for_jnsqs = 0;
	dataset->n_classes		  = n_classes;
	dataset->n_observations	  = n_observations;

	uint32_t total_bits = dataset->n_attributes + dataset->n_bits_for_class;
	uint32_t n_words = total_bits / WORD_BITS + (total_bits % WORD_BITS != 0);

	dataset->n_words = n_words;

	return OK;
}

oknok_t hdf5_read_attribute(hid_t dataset_id, const char* attribute,
							hid_t datatype, void* value)
{
	herr_t status = H5Aexists(dataset_id, attribute);
	if (status < 0)
	{
		// Error reading attribute
		fprintf(stderr, "Error reading attribute %s", attribute);
		return NOK;
	}

	if (status == 0)
	{
		// Attribute does not exist
		fprintf(stderr, "Attribute %s does not exist", attribute);
		return NOK;
	}

	// Open the attribute
	hid_t attr = H5Aopen(dataset_id, attribute, H5P_DEFAULT);
	if (attr < 0)
	{
		fprintf(stderr, "Error opening the attribute %s", attribute);
		return NOK;
	}

	// Read the attribute value
	status = H5Aread(attr, datatype, value);
	if (status < 0)
	{
		fprintf(stderr, "Error reading attribute %s", attribute);
		return NOK;
	}

	// Close the attribute
	status = H5Aclose(attr);
	if (status < 0)
	{
		fprintf(stderr, "Error closing the attribute %s", attribute);
		return NOK;
	}

	return OK;
}

oknok_t hdf5_read_dataset_data(hid_t dataset_id, word_t* data)
{
	// Fill dataset from hdf5 file
	herr_t status = H5Dread(dataset_id, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL,
							H5P_DEFAULT, data);

	if (status < 0)
	{
		fprintf(stderr, "Error reading the dataset data\n");

		data = NULL;
		return NOK;
	}

	return OK;
}

oknok_t hdf5_read_line(const dataset_hdf5_t* dataset, const uint32_t index,
					   const uint32_t n_words, word_t* line)
{
	return hdf5_read_lines(dataset, index, n_words, 1, line);
}

oknok_t hdf5_read_lines(const dataset_hdf5_t* dataset, const uint32_t index,
						const uint32_t n_words, const uint32_t n_lines,
						word_t* lines)
{
	// Setup offset
	hsize_t offset[2] = { index, 0 };

	// Setup count
	hsize_t count[2] = { n_lines, n_words };

	const hsize_t dimensions[2] = { n_lines, n_words };

	// Create a memory dataspace to indicate the size of our buffer/chunk
	hid_t memspace_id = H5Screate_simple(2, dimensions, NULL);

	// Setup line dataspace
	hid_t dataspace_id = H5Dget_space(dataset->dataset_id);

	// Select hyperslab on file dataset
	H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, NULL, count,
						NULL);

	// Read line from dataset
	H5Dread(dataset->dataset_id, H5T_NATIVE_UINT64, memspace_id, dataspace_id,
			H5P_DEFAULT, lines);

	H5Sclose(dataspace_id);
	H5Sclose(memspace_id);

	return OK;
}

oknok_t hdf5_write_attribute(hid_t dataset_id, const char* attribute,
							 hid_t datatype, const void* value)
{
	hid_t attr_dataspace = H5Screate(H5S_SCALAR);
	hid_t attr = H5Acreate(dataset_id, attribute, datatype, attr_dataspace,
						   H5P_DEFAULT, H5P_DEFAULT);
	if (attr < 0)
	{
		fprintf(stderr, "Error cretaing attribute %s.\n", attribute);
		return NOK;
	}

	// Write the attribute to the dataset
	herr_t status = H5Awrite(attr, datatype, value);
	if (status < 0)
	{
		fprintf(stderr, "Error writing attribute %s.\n", attribute);
		return NOK;
	}

	// Close the attribute.
	status = H5Aclose(attr);
	if (status < 0)
	{
		fprintf(stderr, "Error closing attribute %s.\n", attribute);
		return NOK;
	}

	// Close the dataspace.
	status = H5Sclose(attr_dataspace);
	if (status < 0)
	{
		fprintf(stderr, "Error closing attribute %s datatspace.\n", attribute);
		return NOK;
	}

	return OK;
}

void hdf5_get_dataset_dimensions(hid_t dataset_id, hsize_t* dataset_dimensions)
{
	// Get filespace handle first.
	hid_t dataset_space_id = H5Dget_space(dataset_id);

	// Get dataset dimensions.
	H5Sget_simple_extent_dims(dataset_space_id, dataset_dimensions, NULL);

	// Close dataspace
	H5Sclose(dataset_space_id);
}

void hdf5_close_dataset(dataset_hdf5_t* dataset)
{
	H5Dclose(dataset->dataset_id);
	H5Fclose(dataset->file_id);
}
