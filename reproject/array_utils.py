import numpy as np

__all__ = ['map_coordinates']


def pad_edge_1(array):
    return np.pad(array, 1, mode='edge')


def map_coordinates(image, coords, **kwargs):

    # In the built-in scipy map_coordinates, the values are defined at the
    # center of the pixels. This means that map_coordinates does not
    # correctly treat pixels that are in the outer half of the outer pixels.
    # We solve this by extending the array, updating the pixel coordinates,
    # then getting rid of values that were sampled in the range -1 to -0.5
    # and n to n - 0.5.

    from scipy.ndimage import map_coordinates as scipy_map_coordinates

    original_shape = image.shape

    image = pad_edge_1(image)

    map_coords_func = kwargs.pop('map_coords_func', None)

    # By default, use scipy
    if map_coords_func is None:
        values = scipy_map_coordinates(image, coords + 1, **kwargs)
    else:
        print(f'Using {map_coords_func} to do the reprojection')
        # The output option may not be supported by other implementations
        # This could be at the cost of large memory usage and we should
        # deal with this in a more elegant way at some point
        values = kwargs.pop('output')
        values = map_coords_func(image, coords + 1, **kwargs)

    reset = np.zeros(coords.shape[1], dtype=bool)

    for i in range(coords.shape[0]):
        reset |= (coords[i] < -0.5)
        reset |= (coords[i] > original_shape[i] - 0.5)

    values[reset] = kwargs.get('cval', 0.)

    return values
