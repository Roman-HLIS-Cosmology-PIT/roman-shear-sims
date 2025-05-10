from tqdm import tqdm

import numpy as np

import h5py

import galsim
import galsim.roman as roman

from roman_shear_sims.psf_makers import PSFMaker
from roman_shear_sims.wcs import get_SCA_WCS, make_oversample_local_wcs
from roman_shear_sims.constant import WORLD_ORIGIN


if __name__ == "__main__":
    psf_type = "roman"
    chromatic = True

    avg_gal_sed_path = (
        "/Users/aguinot/Documents/roman/test_metacoadd/gal_avg_nz_sed.npy"
    )
    path_names = [
        "/Users/aguinot/Documents/roman/test_metacoadd/psf_roman_vega.hdf5",
        "/Users/aguinot/Documents/roman/test_metacoadd/psf_roman_gal.hdf5",
    ]

    psf_maker = PSFMaker(
        psf_type=psf_type,
        chromatic=chromatic,
        n_waves=5,
    )

    sed_vega = galsim.SED("vega.txt", "nm", "flambda")

    sed_wave, avg_gal_sed_arr = np.load(avg_gal_sed_path)
    sed_lt = galsim.LookupTable(
        sed_wave, avg_gal_sed_arr, interpolant="linear"
    )
    sed_gal = galsim.SED(sed_lt, wave_type="nm", flux_type="fnu")

    for sed_ind, sed_ in tqdm(
        enumerate([sed_vega, sed_gal]), total=2, desc="SED"
    ):
        with h5py.File(path_names[sed_ind], "w") as f:
            for filter_ in tqdm(
                ["Y106", "J129", "H158"],
                total=3,
                leave=False,
                desc="Filter",
            ):
                grp = f.create_group(filter_)
                bp = roman.getBandpasses()[filter_]
                psf_maker.init_psf(band=filter_)
                star = galsim.DeltaFunction()
                star *= sed_.withFlux(1.0, bp)

                for sca in tqdm(
                    range(1, roman.n_sca + 1),
                    total=roman.n_sca,
                    leave=False,
                    desc="SCA",
                ):
                    wcs = get_SCA_WCS(
                        WORLD_ORIGIN,
                        sca,
                        0.0,
                        301,
                    )
                    wcs_oversampled = make_oversample_local_wcs(
                        wcs,
                        WORLD_ORIGIN,
                        6,
                    )
                    psf_ = psf_maker.get_psf(
                        sca=sca,
                        image_pos=galsim.PositionD(
                            roman.n_pix / 2, roman.n_pix / 2
                        ),
                        wcs=wcs,
                    )
                    psf = galsim.Convolve(star, psf_)

                    if sed_ind == 0:
                        wcs_loc = wcs.local(world_pos=WORLD_ORIGIN)
                        pix = wcs_loc.toWorld(galsim.Pixel(1))
                        psf = galsim.Convolve(psf, pix)
                    img = psf.drawImage(
                        nx=301,
                        ny=301,
                        wcs=wcs_oversampled,
                        bandpass=bp,
                        method="no_pixel",
                    ).array
                    grp.create_dataset(str(sca), data=img)
