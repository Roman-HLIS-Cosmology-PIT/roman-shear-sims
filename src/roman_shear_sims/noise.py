import numpy as np

import galsim
from galsim import roman


def make_roman_noise(
    noise_image,
    bandpass,
    exp_time,
    world_pos,
    galsim_rng,
    mjd=None,
    image_factor=1.0,
):
    """
    Add noise to the image based on the Roman characteristics.
    This include:
    - Sky noise (zodial light, thermal background)
    - Dark current noise
    - Read noise
    - Gain

    Parameters
    ----------
    noise_image : galsim.Image
        The image to which noise will be added.
    bandpass : galsim.Bandpass
        The bandpass for which the noise is calculated.
    exp_time : float
        The exposure time in seconds.
    world_pos : galsim.CelestialCoord
        The celestial coordinates of the image center.
    galsim_rng : galsim.BaseDeviate
        The random number generator to use for noise generation.
    mjd : float, optional
        The modified Julian date for the observation. Default: None.
    image_factor : float, optional
        A factor to scale the noise level. Default: 1.0.
    """
    # Sky
    sky_level = roman.getSkyLevel(bandpass, world_pos=world_pos, date=mjd)
    sky_level *= roman.pixel_scale**2
    sky_level += roman.thermal_backgrounds[bandpass.name] * exp_time
    sky_noise = galsim.PoissonNoise(
        rng=galsim_rng, sky_level=sky_level / image_factor
    )
    noise_image.addNoise(sky_noise)
    noise_image.quantize()

    # Dark noise
    dark_curr_lvl = roman.dark_current * exp_time
    dark_noise = galsim.PoissonNoise(
        rng=galsim_rng, sky_level=dark_curr_lvl / image_factor
    )
    noise_image.addNoise(dark_noise)

    # Read noise
    read_noise = galsim.GaussianNoise(
        rng=galsim_rng, sigma=roman.read_noise / np.sqrt(image_factor)
    )
    noise_image.addNoise(read_noise)

    # Apply gain
    noise_image /= roman.gain
    noise_image.quantize()


def make_simple_noise(noise_image, sigma, galsim_rng):
    """
    Add simple Gaussian noise to the image.

    Parameters
    ----------
    noise_image : galsim.Image
        The image to which noise will be added.
    sigma : float
        The standard deviation of the Gaussian noise.
    galsim_rng : galsim.BaseDeviate
        The random number generator to use for noise generation.
    """
    gauss_noise = galsim.GaussianNoise(rng=galsim_rng, sigma=sigma)
    noise_image.addNoise(gauss_noise)

    # Apply gain
    noise_image /= roman.gain
