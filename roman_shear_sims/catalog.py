import numpy as np

import galsim
from galsim import roman

from .constant import ROMAN_N_EFF, DEFAULT_HLR, DEFAULT_MAG, COMMON_ZERO_POINT


LAYOUT_KINDS = ["grid", "random"]
GAL_TYPES = ["gauss", "exp", "dev"]


class SimpleGalaxyCatalog:
    """
    A simple galaxy catalog with a single galaxy type.

    Parameters
    ----------
    img_size : int
        Image size in pixels.
    rng : np.random.RandomState
        Random number generator.
    gal_type : str
        Galaxy type, one of ['gauss', 'exp', 'dev'].
    hlr : float
        Half-light radius of the galaxy.
    mag : float
        Magnitude of the galaxy.
    layout_kind : str
        Layout kind, one of ['grid', 'random'].
    buffer : float
        Buffer size in pixels.
    spacing : float
        Spacing between galaxies in pixels.
    n_gal : int
        Number of galaxies.
    exp_time : float
        Exposure time in seconds.
    """

    def __init__(
        self,
        img_size,
        rng,
        gal_type="exp",
        hlr=DEFAULT_HLR,
        mag=DEFAULT_MAG,
        layout_kind="grid",
        buffer=0.0,
        spacing=5.0,
        n_gal=None,
        exp_time=roman.exptime,
        chromatic=False,
        gal_sed_path=None,
    ):
        self.img_size = img_size
        self.rng = rng
        if gal_type not in GAL_TYPES:
            raise ValueError(
                f"gal_type must be one of {GAL_TYPES}, got {gal_type}"
            )
        self.gal_type = gal_type

        self._hlr = hlr
        self._mag = mag
        self._exp_time = exp_time

        if layout_kind not in LAYOUT_KINDS:
            raise ValueError(
                f"layout_kind must be one of {LAYOUT_KINDS}, got {layout_kind}"
            )
        self.layout_kind = layout_kind
        self.buffer = buffer
        self.spacing = spacing
        self.n_gal = n_gal

        self._chromatic = chromatic
        self._init_catalog(chromatic, gal_sed_path)

    def getNObjects(self):
        return len(self.dx)

    def getObjList(self, band="Y106"):
        bp = roman.getBandpasses()[band]
        zp = self._get_zeropoint(bp)
        objlist = {
            "gsobject": [],
            "dx": [],
            "dy": [],
        }
        for i in range(self.getNObjects()):
            flux = self.get_flux(i, zp)
            if self._chromatic:
                gsobject = self.get_gsobject(i).withFlux(flux, bandpass=bp)
            else:
                gsobject = self.get_gsobject(i).withFlux(flux)
            objlist["gsobject"].append(gsobject)
            objlist["dx"].append(self.dx[i])
            objlist["dy"].append(self.dy[i])
        return objlist

    def _init_catalog(self, chromatic=False, gal_sed_path=None):
        self.dx, self.dy = get_simple_pos(
            self.img_size,
            self.rng,
            layout_kind=self.layout_kind,
            buffer=self.buffer,
            spacing=self.spacing,
            n_gal=self.n_gal,
        )

        if chromatic:
            gal_sed = self._get_sed(gal_sed_path)
        else:
            gal_sed = 1

        self._set_gsobject(gal_sed)
        self._set_gsobject_delta(gal_sed)

    def get_gsobject(self, index):
        return self.gsobject

    def get_gsobject_delta(self):
        return self.gsobject_delta

    def get_flux(self, index, zp):
        return 10 ** (-0.4 * (self._mag - zp))

    def _set_gsobject(self, gal_sed=1):
        if self.gal_type == "gauss":
            self.gsobject = galsim.Gaussian(half_light_radius=self._hlr)
        elif self.gal_type == "exp":
            self.gsobject = galsim.Exponential(half_light_radius=self._hlr)
        elif self.gal_type == "dev":
            self.gsobject = galsim.DeVaucouleurs(half_light_radius=self._hlr)

        self.gsobject *= gal_sed

    def _set_gsobject_delta(self, gal_sed=1):
        self.gsobject_delta = galsim.DeltaFunction()
        self.gsobject_delta *= gal_sed

    def _get_zeropoint(self, bp):
        return (
            bp.zeropoint
            + 2.5 * np.log10(self._exp_time * roman.collecting_area)
            - 2.5 * np.log10(roman.gain)
        )

    def get_flux_scaling(self, band, target_zp=COMMON_ZERO_POINT):
        bp = roman.getBandpasses()[band]
        current_zp = self._get_zeropoint(bp)
        zp_diff = target_zp - current_zp
        return 10 ** (0.4 * zp_diff)

    def _get_sed(self, gal_sed_path):
        sed_wave, avg_gal_sed_arr = np.load(gal_sed_path)
        sed_lt = galsim.LookupTable(
            sed_wave, avg_gal_sed_arr, interpolant="linear"
        )
        sed = galsim.SED(sed_lt, wave_type="nm", flux_type="fnu")
        return sed


def get_simple_pos(
    img_size,
    rng,
    layout_kind="grid",
    buffer=0.0,
    spacing=5.0,
    n_gal=None,
):
    """
    Those functions are taken from the descwl-shear-sims package,
    refs: https://github.com/LSSTDESC/descwl-shear-sims/blob/master/descwl_shear_sims/layout/shifts.py
    """
    pixel_scale = roman.pixel_scale

    img_size = img_size
    buff_img_size = img_size - 2 * buffer
    img_size_world = img_size * pixel_scale
    buff_img_size_world = buff_img_size * pixel_scale

    buffer_world = buffer * pixel_scale

    if layout_kind not in LAYOUT_KINDS:
        raise ValueError(
            f"layout_kind must be one of {LAYOUT_KINDS}, got {layout_kind}"
        )
    layout_kind = layout_kind
    buffer = buffer
    spacing = spacing

    if layout_kind == "grid":
        n_obj_side = int(np.floor((img_size_world) / spacing))
        if n_obj_side == 0:
            n_obj_side = 1

        x = spacing * (np.arange(n_obj_side) - (n_obj_side - 1) / 2)
        msk = (x >= -(img_size_world / 2 - buffer_world)) & (
            x <= img_size_world / 2 - buffer_world
        )
        x = x[msk]

        xx, yy = np.meshgrid(x, x)
        xx = xx.flatten()
        yy = yy.flatten()

        xx += rng.uniform(-pixel_scale / 2, pixel_scale / 2.0, len(xx))
        yy += rng.uniform(-pixel_scale / 2, pixel_scale / 2.0, len(yy))

    elif layout_kind == "random":
        if n_gal is None:
            area = buff_img_size_world**2
            area /= 3600
            n_gal = int(np.floor(area * ROMAN_N_EFF))

        xx = rng.uniform(
            low=buffer_world,
            high=buff_img_size_world + buffer_world,
            size=n_gal,
        )
        yy = rng.uniform(
            low=buffer_world,
            high=buff_img_size_world + buffer_world,
            size=n_gal,
        )

        xx -= img_size_world / 2.0
        yy -= img_size_world / 2.0

    if n_gal is None and layout_kind == "grid":
        n_gal = len(xx)

    return xx[:n_gal], yy[:n_gal]
