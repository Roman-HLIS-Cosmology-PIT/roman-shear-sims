from IPython import get_ipython

if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
from memory_profiler import profile
import gc

# import concurrent.futures
from .executor_utils import get_executor

import math

import numpy as np

import galsim
from galsim import roman

from .utils import random_ra_dec_in_healpix
from .noise import make_roman_noise, make_simple_noise
from .wcs import (
    get_SCA_WCS,
    make_oversample_local_wcs,
    make_simple_exp_wcs,
    make_simple_coadd_wcs,
)
from .constant import WORLD_ORIGIN


# @profile
def make_sim(
    rng,
    galaxy_catalog,
    psf_maker,
    n_epochs=6,
    exp_time=107,
    cell_size_pix=500,
    oversamp_factor=3,
    # stamp_size=150,
    bands=["Y106", "J129", "H158"],
    g1=np.array([0.0]),
    g2=np.array([0.0]),
    chromatic=False,
    simple_noise=False,
    noise_sigma=None,
    draw_method="phot",
    avg_gal_sed_path="/Users/aguinot/Documents/roman/test_metacoadd/gal_avg_nz_sed.npy",
    make_child_process=False,
    make_deconv_psf=True,
    make_true_psf=True,
    verbose=True,
):
    # Set center
    # cell_center_ra, cell_center_dec = random_ra_dec_in_healpix(rng, 32, 10307)

    # cell_center_world = galsim.CelestialCoord(
    #     cell_center_ra * galsim.degrees, cell_center_dec * galsim.degrees
    # )
    cell_center_world = WORLD_ORIGIN

    # Prepare g1, g2
    g1 = np.atleast_1d(g1)
    g2 = np.atleast_1d(g2)

    # Band loop
    final_dict = {}
    for band in tqdm(bands, desc="Band loop", disable=not verbose):
        epoch_list = []

        psf_maker.init_psf(band=band)
        pa = rng.uniform(low=150, high=190)

        # Epoch loop
        for i in tqdm(
            range(n_epochs),
            desc="Epoch loop",
            leave=False,
            disable=not verbose,
        ):
            if not make_child_process:
                epoch_dict = make_exp(
                    rng,
                    galaxy_catalog,
                    psf_maker,
                    band,
                    cell_center_world,
                    g1,
                    g2,
                    pa_point=pa,
                    exp_time=exp_time,
                    cell_size_pix=cell_size_pix,
                    oversamp_factor=oversamp_factor,
                    chromatic=chromatic,
                    simple_noise=simple_noise,
                    noise_sigma=noise_sigma,
                    draw_method=draw_method,
                    avg_gal_sed_path=avg_gal_sed_path,
                    make_deconv_psf=make_deconv_psf,
                    make_true_psf=make_true_psf,
                    verbose=verbose,
                )
            else:
                epoch_dict = run_in_child_process(
                    make_exp,
                    rng,
                    galaxy_catalog,
                    psf_maker,
                    band,
                    cell_center_world,
                    g1,
                    g2,
                    pa_point=pa,
                    exp_time=exp_time,
                    cell_size_pix=cell_size_pix,
                    oversamp_factor=oversamp_factor,
                    chromatic=chromatic,
                    simple_noise=simple_noise,
                    noise_sigma=noise_sigma,
                    draw_method=draw_method,
                    avg_gal_sed_path=avg_gal_sed_path,
                    make_deconv_psf=make_deconv_psf,
                    make_true_psf=make_true_psf,
                    verbose=verbose,
                )
            epoch_dict["cell_center_world"] = cell_center_world
            epoch_list.append(epoch_dict)
        final_dict[band] = epoch_list

    return final_dict


def run_in_child_process(func, *args, **kwargs):
    # with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
    #     future = executor.submit(func, *args, **kwargs)
    # return future.result()
    executor = get_executor()
    result = executor.submit(func, *args, **kwargs).result()
    return result


# @profile
def make_exp(
    rng,
    galaxy_catalog,
    psf_maker,
    band,
    cell_center_world,
    g1,
    g2,
    pa_point=0.0,
    exp_time=107,
    cell_size_pix=500,
    oversamp_factor=3,
    chromatic=False,
    simple_noise=False,
    noise_sigma=None,
    draw_method="phot",
    avg_gal_sed_path=None,
    make_deconv_psf=False,
    make_true_psf=False,
    verbose=True,
):
    bp_ = roman.getBandpasses()[band]
    seed_epoch = rng.randint(0, 2**32)

    sca = rng.randint(1, roman.n_sca + 1)
    epoch_dict = {
        "sca": sca,
    }
    exp_center = cell_center_world
    wcs = get_SCA_WCS(
        # cell_center_world,
        exp_center,
        sca,
        # PA=pa_point,
        PA=0.0,
        img_size=cell_size_pix,
    )
    # wcs = make_simple_exp_wcs(
    #     cell_center_world,
    #     PA=0.0,
    #     img_size=cell_size_pix,
    # )
    # wcs = make_simple_coadd_wcs(
    #     cell_center_world,
    #     img_size=cell_size_pix,
    # )
    epoch_dict["wcs"] = wcs
    epoch_dict["flux_scaling"] = galaxy_catalog.get_flux_scaling(band)

    psf = psf_maker.get_psf(
        sca=sca,
        image_pos=galsim.PositionD(roman.n_pix / 2, roman.n_pix / 2),
        wcs=wcs,
    )

    ## Make noise
    noise_img = galsim.Image(cell_size_pix, cell_size_pix, wcs=wcs)
    rng_galsim = galsim.BaseDeviate(seed_epoch)
    if not simple_noise:
        make_roman_noise(
            noise_img,
            bp_,
            exp_time,
            # cell_center_world,
            exp_center,
            rng_galsim,
        )
    else:
        if noise_sigma is None:
            raise ValueError("noise_sigma must be provided")
        make_simple_noise(
            noise_img,
            noise_sigma,
            rng_galsim,
        )

    # Make avg PSF used for deconvolution
    wcs_oversampled = None
    if make_deconv_psf:
        wcs_oversampled = make_oversample_local_wcs(
            wcs,
            cell_center_world,
            oversamp_factor,
        )
        # psf_img_deconv, wcs_oversampled, psf_obj_deconv = get_deconv_psf(
        psf_img_deconv, wcs_oversampled = get_deconv_psf(
            psf,
            wcs,
            wcs_oversampled,
            # cell_center_world,
            exp_center,
            bp=bp_,
            oversamp_factor=oversamp_factor,
            chromatic=chromatic,
            # avg_gal_sed_path=avg_gal_sed_path,
            avg_gal_sed_path=None,
        )
        epoch_dict["psf_avg"] = psf_img_deconv
        # epoch_dict["psf_avg_galsim"] = psf_obj_deconv
        epoch_dict["wcs_oversampled"] = wcs_oversampled
    else:
        epoch_dict["psf_avg"] = None
        epoch_dict["psf_avg_galsim"] = None

    # Make true PSF
    if chromatic & make_true_psf:
        if wcs_oversampled is None:
            wcs_oversampled = make_oversample_local_wcs(
                wcs,
                cell_center_world,
                oversamp_factor,
            )
        epoch_dict["psf_true_galsim"] = get_true_psf(
            galaxy_catalog.get_gsobject_delta().withFlux(1, bp_),
            psf,
            wcs_oversampled,
            bp_,
        )
    else:
        epoch_dict["psf_true_galsim"] = None
    epoch_dict["wcs_oversampled"] = wcs_oversampled

    # n_obj = galaxy_catalog.getNObjects()
    objlist = galaxy_catalog.getObjList(bandpass=bp_)
    n_obj = len(objlist["gsobject"])

    epoch_dict["sci"] = {}
    for g1_, g2_ in zip(g1, g2):
        # Make image
        final_img = galsim.Image(cell_size_pix, cell_size_pix, wcs=wcs)

        # Object loop
        for obj_ind in tqdm(
            range(n_obj),
            total=n_obj,
            leave=False,
            desc="Obj loop",
            disable=not verbose,
        ):
            # Make obj
            gal = objlist["gsobject"][obj_ind]
            dx = objlist["dx"][obj_ind]
            dy = objlist["dy"][obj_ind]

            gal = gal.shear(g1=g1_, g2=g2_)
            obj = galsim.Convolve(gal, psf)

            stamp_image = get_stamp(
                obj,
                dx,
                dy,
                wcs,
                cell_center_world,
                bp_,
                rng_galsim,
                draw_method=draw_method,
            )

            b = stamp_image.bounds & final_img.bounds
            if b.isDefined():
                final_img[b] += stamp_image[b]

        final_img /= roman.gain
        final_img += noise_img
        epoch_dict["sci"][f"shear_{g1_}_{g2_}"] = final_img.array
    epoch_dict["noise"] = noise_img.array
    epoch_dict["noise_var"] = noise_img.array.var()
    epoch_dict["weight"] = (
        np.ones_like(noise_img.array) / noise_img.array.var()
    )

    return epoch_dict


# @profile
def get_stamp(
    obj,
    dx,
    dy,
    wcs,
    cell_center_world,
    bp,
    rng_galsim,
    draw_method="phot",
):
    # gal = objlist["gsobject"][obj_ind]
    # dx = objlist["dx"][obj_ind]
    # dy = objlist["dy"][obj_ind]

    # # Make obj
    # gal = gal.shear(g1=g1, g2=g2)
    # obj = galsim.Convolve([gal, psf])

    world_pos = cell_center_world.deproject(
        dx * galsim.arcsec,
        dy * galsim.arcsec,
    )
    image_pos = wcs.toImage(world_pos)

    # Set center and offset
    nominal_x = image_pos.x + 0.5
    nominal_y = image_pos.y + 0.5

    stamp_center = galsim.PositionI(
        int(math.floor(nominal_x + 0.5)),
        int(math.floor(nominal_y + 0.5)),
    )
    # stamp_offset = galsim.PositionD(
    #     nominal_x - stamp_center.x, nominal_y - stamp_center.y
    # )

    if draw_method == "phot":
        stamp_size = 150
        rng_draw = rng_galsim
        maxN = int(1e6)
    else:
        # stamp_size = obj.getGoodImageSize(roman.pixel_scale)
        # print(stamp_size)
        stamp_size = 100
        rng_draw = None
        maxN = None

    stamp_image = obj.drawImage(
        nx=stamp_size,
        ny=stamp_size,
        bandpass=bp,
        wcs=wcs.local(world_pos=world_pos),
        method=draw_method,
        center=image_pos,
        rng=rng_draw,
        maxN=maxN,
    )
    del obj
    gc.collect()

    return stamp_image


# @profile
def get_true_psf(star, psf, wcs, bp):
    tmp_true_ = galsim.Convolve(
        star,
        psf,
    )

    true_psf_img = tmp_true_.drawImage(
        nx=301,
        ny=301,
        wcs=wcs,
        bandpass=bp,
        method="no_pixel",
        # method="phot",
        # n_photons=1e6,
    )

    # interp_img = galsim.InterpolatedImage(
    #     true_psf_img, x_interpolant="lanczos15"
    # )
    # pix = wcs.toWorld(galsim.Pixel(1))
    # inv_pix = galsim.Deconvolve(pix)
    # interp_img = galsim.Convolve(interp_img, inv_pix)

    # return interp_img
    return true_psf_img.array


def get_deconv_psf(
    psf,
    wcs,
    wcs_oversampled,
    cell_center_world,
    bp=None,
    oversamp_factor=3,
    chromatic=False,
    avg_gal_sed_path=None,
):
    star_psf = galsim.DeltaFunction()

    # Average galaxy SED
    if chromatic:
        if avg_gal_sed_path is None:
            sed = galsim.SED("vega.txt", "nm", "flambda")
            # sed = galsim.SED(
            #     galsim.LookupTable(
            #         [100, 1000, 2000], [0.0, 1.0, 10.0], interpolant="linear"
            #     ),
            #     wave_type="nm",
            #     flux_type="flambda",
            # )
            # avg_star_sed_path = "/Users/aguinot/Documents/roman/test_metacoadd/star_avg_sed.npz"
            # sed_wave, avg_star_sed_arr = (
            #     np.load(avg_star_sed_path)["x"],
            #     np.load(avg_star_sed_path)["y"],
            # )
            # sed_lt = galsim.LookupTable(
            #     sed_wave, avg_star_sed_arr, interpolant="linear"
            # )
            # sed = galsim.SED(sed_lt, wave_type="nm", flux_type="flambda")
        else:
            sed_wave, avg_gal_sed_arr = np.load(avg_gal_sed_path)
            sed_lt = galsim.LookupTable(
                sed_wave, avg_gal_sed_arr, interpolant="linear"
            )
            sed = galsim.SED(sed_lt, wave_type="nm", flux_type="fnu")
        star_psf *= sed.withFlux(1, bp)

    wcs_local_center = wcs.local(world_pos=cell_center_world)
    pixel_ori = wcs_local_center.toWorld(galsim.Pixel(1))
    psf_obj = galsim.Convolve(psf, star_psf)
    psf_obj = galsim.Convolve(psf_obj, pixel_ori)
    psf_img = psf_obj.drawImage(
        nx=301,
        ny=301,
        wcs=wcs_oversampled,
        bandpass=bp,
        method="no_pixel",
        # method="phot",
        # n_photons=1e6,
        # rng=galsim.BaseDeviate(42),
    )

    return psf_img.array, wcs_oversampled  # , psf_obj
