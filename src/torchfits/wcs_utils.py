from .fits_reader_cpp import world_to_pixel as _world_to_pixel
from .fits_reader_cpp import pixel_to_world as _pixel_to_world

def world_to_pixel(world_coords, header):
    """
    Convert world coordinates to pixel coordinates.
    
    Args:
        world_coords (torch.Tensor): World coordinates (N,M) where M is the number of dimensions
        header (dict): FITS header with WCS information
        
    Returns:
        tuple: (pixel_coords, status) where:
            - pixel_coords (torch.Tensor): Pixel coordinates (N,M)
            - status (torch.Tensor): Status values for each conversion (N)
    """
    return _world_to_pixel(world_coords, header)

def pixel_to_world(pixel_coords, header):
    """
    Convert pixel coordinates to world coordinates.
    
    Args:
        pixel_coords (torch.Tensor): Pixel coordinates (N,M) where M is the number of dimensions
        header (dict): FITS header with WCS information
        
    Returns:
        tuple: (world_coords, status) where:
            - world_coords (torch.Tensor): World coordinates (N,M)
            - status (torch.Tensor): Status values for each conversion (N)
    """
    return _pixel_to_world(pixel_coords, header)
