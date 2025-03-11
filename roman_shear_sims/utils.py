import numpy as np

import healpy as hp


def random_ra_dec_in_healpix(rng, nside, pixel_index):
    """
    Generate a random (RA, Dec) coordinate uniformly within a given HEALPix pixel.

    Parameters:
        seed (int): Random seed for reproducibility.
        nside (int): The nside parameter of the HEALPix grid.
        pixel_index (int): The HEALPix pixel index.

    Returns:
        tuple: (RA, Dec) in degrees.
    """

    # Generate a random point inside the HEALPix pixel using uniform sampling
    theta, phi = hp.pix2ang(nside, pixel_index)
    pixel_area = hp.nside2pixarea(nside)

    while True:
        dtheta = rng.uniform(-np.sqrt(pixel_area), np.sqrt(pixel_area))
        dphi = rng.uniform(
            -np.sqrt(pixel_area), np.sqrt(pixel_area) / np.sin(theta)
        )

        new_theta = theta + dtheta
        new_phi = phi + dphi

        if hp.pixelfunc.ang2pix(nside, new_theta, new_phi) == pixel_index:
            break

    # Convert to RA, Dec (degrees)
    ra = np.degrees(new_phi)  # Phi is the RA
    dec = 90 - np.degrees(new_theta)  # Theta is measured from the north pole

    return ra, dec
