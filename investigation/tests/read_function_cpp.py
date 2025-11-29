# Working C++ backend implementation for torchfits.read()
# To be copied into __init__.py replacing the astropy implementation

def read_cpp_backend():
    """
    This is the working C++ backend code to replace lines 136-267 in __init__.py
    """
    code = '''
    # Cache tracking
    global _cache_stats, _file_cache
    _cache_stats['total_requests'] += 1

    cache_key = f"{path}:{hdu}:{device}:{fp16}:{bf16}:{columns}:{start_row}:{num_rows}"
    if cache_key in _file_cache:
        _cache_stats['hits'] += 1
        cached_data, cached_header = _file_cache[cache_key]
        if isinstance(cached_data, torch.Tensor) and device != 'cpu':
            cached_data = cached_data.to(device)
        return cached_data, cached_header
    else:
        _cache_stats['misses'] += 1

    try:
        # Open FITS file with C++ backend
        file_handle = cpp.open_fits_file(path, "r")

        try:
            # Handle HDU selection (int or name)
            if isinstance(hdu, str):
                num_hdus = cpp.get_num_hdus(file_handle)
                hdu_num = None
                for i in range(num_hdus):
                    try:
                        hdr = cpp.read_header(file_handle, i)
                        if hdr.get('EXTNAME') == hdu:
                            hdu_num = i
                            break
                    except:
                        continue
                if hdu_num is None:
                    raise ValueError(f"HDU '{hdu}' not found in file")
            else:
                hdu_num = hdu

            # Get header
            header = cpp.read_header(file_handle, hdu_num)

            # Try to read as IMAGE (most common, skip get_hdu_type for speed)
            try:
                data = cpp.read_full(file_handle, hdu_num)

                # Apply precision conversion
                if fp16:
                    data = data.to(torch.float16)
                elif bf16:
                    data = data.to(torch.bfloat16)

                # Move to device
                if device != 'cpu':
                    data = data.to(device)

                # Cache result
                _file_cache[cache_key] = (data.cpu() if device != 'cpu' else data, header)
                _cache_stats['cache_size'] = len(_file_cache)

                return data, header

            except (RuntimeError, TypeError):
                # Not an image, try table (with astropy fallback for now)
                from astropy.io import fits as astropy_fits
                with astropy_fits.open(path) as hdul:
                    table_data = {}
                    column_names = columns if columns else [col.name for col in hdul[hdu_num].columns]

                    for col_name in column_names:
                        try:
                            col_data = hdul[hdu_num].data[col_name]
                            if start_row > 1 or num_rows != -1:
                                end_row = start_row + num_rows - 1 if num_rows != -1 else len(col_data)
                                col_data = col_data[start_row-1:end_row]

                            if col_data.dtype.kind in ['U', 'S']:
                                continue

                            if col_data.dtype.kind in ['i', 'u']:
                                if col_data.dtype.itemsize <= 1:
                                    numpy_dtype = np.int8 if col_data.dtype.kind == 'i' else np.uint8
                                elif col_data.dtype.itemsize <= 2:
                                    numpy_dtype = np.int16
                                elif col_data.dtype.itemsize <= 4:
                                    numpy_dtype = np.int32
                                else:
                                    numpy_dtype = np.int64
                                table_data[col_name] = torch.from_numpy(col_data.astype(numpy_dtype))
                            else:
                                if col_data.dtype == np.float64:
                                    table_data[col_name] = torch.from_numpy(col_data)
                                else:
                                    table_data[col_name] = torch.from_numpy(col_data.astype(np.float32))
                        except Exception as e:
                            print(f'Warning: Failed to read column {col_name}: {e}')
                            continue

                    _file_cache[cache_key] = (table_data, header)
                    _cache_stats['cache_size'] = len(_file_cache)
                    return table_data, header

        finally:
            try:
                cpp.close_fits_file(file_handle)
            except:
                pass

    except Exception as e:
        raise RuntimeError(f"Failed to read FITS file '{path}': {e}") from e
'''
    return code

if __name__ == "__main__":
    print(read_cpp_backend())
