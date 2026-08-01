#pragma once

#include <string>
#include <vector>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <fitsio.h>

void write_fits_table(const char* filename, nb::dict tensor_dict, nb::dict header,
                      bool overwrite, nb::object schema_obj, const std::string& table_type);
long infer_num_rows_from_payload(nb::dict tensor_dict);
void append_rows(const char* filename, int hdu_num, nb::dict tensor_dict);
void insert_rows(const char* filename, int hdu_num, nb::dict tensor_dict, long start_row);
void delete_rows(const char* filename, int hdu_num, long start_row, long num_rows);
void update_rows(const char* filename, int hdu_num, nb::dict tensor_dict,
                 long start_row, long num_rows);
void update_rows_mmap(const char* filename, int hdu_num, nb::dict tensor_dict,
                      long start_row, long num_rows);
void rename_columns(const char* filename, int hdu_num, nb::dict mapping);
void drop_columns(const char* filename, int hdu_num, nb::list columns);
