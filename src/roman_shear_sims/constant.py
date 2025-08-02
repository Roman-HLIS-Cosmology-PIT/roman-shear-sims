import galsim
from galsim import roman

COMMON_ZERO_POINT = 30

SCALE = roman.pixel_scale

ROMAN_N_EFF = 41.0

DEFAULT_HLR = 0.5
DEFAULT_MAG = 25

MJD = 61937.090308
WORLD_ORIGIN = galsim.CelestialCoord(
    ra=10.161290322580646 * galsim.degrees,
    dec=-43.40685848593699 * galsim.degrees,
)

# IMCOM constants
IMCOM_BLOCK_SIZE = 2688
IMCOM_PIXEL_SCALE = 0.0390625
IMCOM_PSF_FWHM = {
    "Y106": 0.22,
    "J129": 0.231,
    "H158": 0.242,
    "F184": 0.253,
    "K213": 0.264,
}
