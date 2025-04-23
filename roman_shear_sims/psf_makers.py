import numpy as np

import galsim
from galsim import roman

PSF_TYPES = ["gauss", "airy", "obs_airy", "roman"]

_rad_to_arcsec = 180 * 3600 / np.pi


class PSFMaker:
    def __init__(
        self,
        psf_type="gauss",
        fwhm=None,
        chromatic=True,
        wave=None,
        sca=1,
        pupil_bin=8,
        n_waves=10,
        image_pos=None,
    ):
        if psf_type not in PSF_TYPES:
            raise ValueError(
                f"psf_type must be one of {PSF_TYPES}, got {psf_type}"
            )
        self.psf_type = psf_type

        self.fwhm = fwhm
        self.sca = sca
        self.pupil_bin = pupil_bin
        self.chromatic = chromatic
        self.wave = wave
        self.n_waves = n_waves

    def init_psf(self, band="Y106"):
        self.band = band
        if self.wave is None:
            bp = roman.getBandpasses()[self.band]
            self.wave = bp.effective_wavelength
        if self.fwhm is None:
            self.fwhm = self.wave / roman.pixel_scale * 1e-9
            self.fwhm *= _rad_to_arcsec

    def get_psf(self, sca=None, image_pos=None, wcs=None):
        if self.psf_type == "gauss":
            psf = galsim.Gaussian(fwhm=self.fwhm)
        elif self.psf_type == "airy":
            psf = galsim.Airy(lam=self.wave, diam=roman.diameter)
        elif self.psf_type == "obs_airy":
            psf = galsim.Airy(
                lam=self.wave,
                diam=roman.diameter,
                obscuration=roman.obscuration,
            )
        elif self.psf_type == "roman":
            if self.chromatic:
                wave = None
            else:
                wave = self.wave
            psf = roman.getPSF(
                SCA=sca,
                bandpass=self.band,
                SCA_pos=image_pos,
                pupil_bin=self.pupil_bin,
                wcs=wcs,
                n_waves=self.n_waves,
                wavelength=wave,
            )
        return psf
