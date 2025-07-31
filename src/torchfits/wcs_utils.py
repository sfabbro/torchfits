import torch
from . import fits_reader_cpp

def world_to_pixel(world_coords, header):
    """
    Converts world coordinates to pixel coordinates.
    """
    return fits_reader_cpp.world_to_pixel(world_coords, header)

def pixel_to_world(pixel_coords, header):
    """
    Converts pixel coordinates to world coordinates.
    """
    return fits_reader_cpp.pixel_to_world(pixel_coords, header)
