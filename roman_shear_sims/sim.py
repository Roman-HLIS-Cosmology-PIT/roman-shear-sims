from IPython import get_ipython

if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

import math

import numpy as np

import galsim
from galsim import roman

from .utils import random_ra_dec_in_healpix
from .catalog import SkyCatalogInterfaceNew
from .noise import make_roman_noise, make_simple_noise
from .wcs import get_SCA_WCS, make_simple_wcs, make_oversample_local_wcs


def make_sim(
    seed,
    mjd,
    n_epochs=6,
    exp_time=107,
    cell_size_pix=500,
    oversamp_factor=3,
    stamp_size=150,
    bands=["Y106", "J129", "H158"],
    g1=[0.02, -0.02],
    g2=[0.0, 0.0],
    simple_noise=False,
    noise_sigma=None,
    skycat_path="/Users/aguinot/Documents/euclid_sim/imsim_data/skyCatalog.yaml",
    avg_gal_sed_path="/Users/aguinot/Documents/roman/test_metacoadd/gal_avg_nz_sed.npy",
    verbose=True,
):
    rng = np.random.RandomState(seed)

    cell_center_ra, cell_center_dec = random_ra_dec_in_healpix(rng, 32, 10307)

    cell_center_world = galsim.CelestialCoord(
        cell_center_ra * galsim.degrees, cell_center_dec * galsim.degrees
    )

    # Average galaxy SED
    sed_wave, avg_gal_sed = np.load(avg_gal_sed_path)
    sed_lt = galsim.LookupTable(sed_wave, avg_gal_sed, interpolant="linear")
    sed = galsim.SED(sed_lt, wave_type="nm", flux_type="fnu")

    wcs_simple = make_simple_wcs(cell_center_world, img_size=cell_size_pix)

    skycat = SkyCatalogInterfaceNew(
        skycat_path,
        exp_time,
        wcs=wcs_simple,
        mjd=mjd,
        xsize=cell_size_pix,
        ysize=cell_size_pix,
        obj_types=[
            "diffsky_galaxy",
        ],
        edge_pix=50,
    )

    n_obj = skycat.getNObjects()

    # Band loop
    final_dict = {}
    for band in tqdm(bands, desc="Band loop", disable=not verbose):
        bp_ = roman.getBandpasses()[band]
        star_psf = galsim.DeltaFunction() * sed.withFlux(1, bp_)
        seed_band = rng.randint(2**32)
        rng_band = np.random.RandomState(seed_band)
        epoch_list = []

        # Epoch loop
        for i in tqdm(
            range(n_epochs),
            desc="Epoch loop",
            leave=False,
            disable=not verbose,
        ):
            seed_epoch = rng_band.randint(2**32)
            rng_epoch = np.random.RandomState(seed_epoch)

            sca = rng_epoch.randint(1, roman.n_sca + 1)
            epoch_dict = {
                "sca": sca,
            }
            wcs = get_SCA_WCS(
                cell_center_world,
                sca,
                PA=0.0,
                img_size=cell_size_pix,
            )
            epoch_dict["wcs"] = wcs

            psf = roman.getPSF(
                sca,
                bp_.name,
                galsim.PositionD(roman.n_pix / 2, roman.n_pix / 2),
                pupil_bin=8,
                wcs=wcs,
                n_waves=16,
            )

            ## Make noise
            noise_img = galsim.Image(cell_size_pix, cell_size_pix, wcs=wcs)
            rng_galsim_noise = galsim.BaseDeviate(seed_epoch)
            if not simple_noise:
                make_roman_noise(
                    noise_img,
                    bp_,
                    exp_time,
                    mjd,
                    cell_center_world,
                    rng_galsim_noise,
                )
            else:
                if noise_sigma is None:
                    raise ValueError("noise_sigma must be provided")
                make_simple_noise(
                    noise_img,
                    noise_sigma,
                    rng_galsim_noise,
                )

            ## Make avg PSF
            wcs_oversampled = make_oversample_local_wcs(
                wcs,
                cell_center_world,
                oversamp_factor,
            )
            psf_obj = galsim.Convolve(psf, star_psf)
            psf_img = psf_obj.drawImage(
                nx=151,
                ny=151,
                wcs=wcs_oversampled,
                bandpass=bp_,
            )
            epoch_dict["psf_avg"] = psf_img.array

            # Shear loop
            for g1_, g2_ in tqdm(
                zip(g1, g2),
                total=len(g1),
                leave=False,
                desc="Shear loop",
                disable=not verbose,
            ):
                rng_galsim = galsim.BaseDeviate(seed_epoch)
                final_img = galsim.Image(cell_size_pix, cell_size_pix, wcs=wcs)

                # Object loop
                for obj_ind in tqdm(
                    range(n_obj),
                    total=n_obj,
                    leave=False,
                    desc="Obj loop",
                    disable=not verbose,
                ):
                    gal = skycat.getObj(obj_ind, rng=rng_galsim, bandpass=bp_)
                    if gal is None:
                        continue

                    world_pos = skycat.getWorldPos(obj_ind)
                    image_pos = wcs.toImage(world_pos)

                    # Set center and offset
                    nominal_x = image_pos.x + 0.5
                    nominal_y = image_pos.y + 0.5

                    stamp_center = galsim.PositionI(
                        int(math.floor(nominal_x + 0.5)),
                        int(math.floor(nominal_y + 0.5)),
                    )
                    stamp_offset = galsim.PositionD(
                        nominal_x - stamp_center.x, nominal_y - stamp_center.y
                    )

                    bounds = galsim.BoundsI(1, stamp_size, 1, stamp_size)
                    bounds = bounds.shift(stamp_center - bounds.center)
                    stamp_image = galsim.Image(bounds=bounds, wcs=wcs)

                    # Make obj
                    gal = gal.shear(g1=g1_, g2=g2_)
                    obj = galsim.Convolve([gal, psf])

                    obj.drawImage(
                        bp_,
                        image=stamp_image,
                        wcs=wcs,
                        method="phot",
                        offset=stamp_offset,
                        rng=rng_galsim,
                        maxN=int(1e6),
                        add_to_image=True,
                    )

                    b = stamp_image.bounds & final_img.bounds
                    if b.isDefined():
                        final_img[b] += stamp_image[b]

                final_img /= roman.gain
                final_img.quantize()
                final_img += noise_img
                epoch_dict[f"img_{g1_}_{g2_}"] = final_img.array
                epoch_dict["noise"] = noise_img.array
                epoch_dict["noise_var"] = noise_img.array.var()
                epoch_dict["weight"] = 1 / noise_img.array.var()
            epoch_list.append(epoch_dict)
        final_dict[band] = epoch_list

    return final_dict
