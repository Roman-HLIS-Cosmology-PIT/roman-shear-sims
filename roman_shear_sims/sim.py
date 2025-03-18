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
from .noise import make_roman_noise, make_simple_noise
from .wcs import (
    get_SCA_WCS,
    make_oversample_local_wcs,
    # make_simple_coadd_wcs,
)


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
    g1=0.0,
    g2=0.0,
    chromatic=True,
    simple_noise=False,
    noise_sigma=None,
    draw_method="phot",
    avg_gal_sed_path="/Users/aguinot/Documents/roman/test_metacoadd/gal_avg_nz_sed.npy",
    verbose=True,
):
    cell_center_ra, cell_center_dec = random_ra_dec_in_healpix(rng, 32, 10307)

    cell_center_world = galsim.CelestialCoord(
        cell_center_ra * galsim.degrees, cell_center_dec * galsim.degrees
    )

    # Band loop
    final_dict = {}
    for band in tqdm(bands, desc="Band loop", disable=not verbose):
        # bp_ = roman.getBandpasses()[band]
        # star_psf = galsim.DeltaFunction()
        # if chromatic:
        # star_psf *= avg_gal_sed.withFlux(1, bp_)
        # wave_psf = None
        # bp_draw = bp_
        # else:
        # wave_psf = bp_.effective_wavelength
        # bp_draw = None
        epoch_list = []

        psf_maker.init_psf(band=band)

        # Epoch loop
        for i in tqdm(
            range(n_epochs),
            desc="Epoch loop",
            leave=False,
            disable=not verbose,
        ):
            epoch_dict = make_exp(
                rng,
                galaxy_catalog,
                psf_maker,
                band,
                cell_center_world,
                g1,
                g2,
                exp_time=exp_time,
                cell_size_pix=cell_size_pix,
                oversamp_factor=oversamp_factor,
                chromatic=chromatic,
                simple_noise=simple_noise,
                noise_sigma=noise_sigma,
                draw_method=draw_method,
                avg_gal_sed_path=avg_gal_sed_path,
                verbose=verbose,
            )
            epoch_dict["cell_center_world"] = cell_center_world
            epoch_list.append(epoch_dict)
        final_dict[band] = epoch_list

    return final_dict


def make_exp(
    rng,
    galaxy_catalog,
    psf_maker,
    band,
    cell_center_world,
    g1,
    g2,
    exp_time=107,
    cell_size_pix=500,
    oversamp_factor=3,
    chromatic=True,
    simple_noise=False,
    noise_sigma=None,
    draw_method="phot",
    avg_gal_sed_path=None,
    verbose=True,
):
    bp_ = roman.getBandpasses()[band]
    seed_epoch = rng.randint(0, 2**32)

    sca = rng.randint(1, roman.n_sca + 1)
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

    psf = psf_maker.get_psf(wcs=wcs, sca=sca)

    ## Make noise
    noise_img = galsim.Image(cell_size_pix, cell_size_pix, wcs=wcs)
    rng_galsim = galsim.BaseDeviate(seed_epoch)
    if not simple_noise:
        make_roman_noise(
            noise_img,
            bp_,
            exp_time,
            cell_center_world,
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

    ## Make avg PSF used for deconvolution
    psf_img_deconv, wcs_oversampled = get_deconv_psf(
        psf,
        wcs,
        cell_center_world,
        bp=bp_,
        oversamp_factor=oversamp_factor,
        chromatic=chromatic,
        avg_gal_sed_path=avg_gal_sed_path,
    )
    epoch_dict["psf_avg"] = psf_img_deconv
    epoch_dict["wcs_oversampled"] = wcs_oversampled

    final_img = galsim.Image(cell_size_pix, cell_size_pix, wcs=wcs)

    n_obj = galaxy_catalog.getNObjects()
    objlist = galaxy_catalog.getObjList()
    # Object loop
    for obj_ind in tqdm(
        range(n_obj),
        total=n_obj,
        leave=False,
        desc="Obj loop",
        disable=not verbose,
    ):
        gal = objlist["gsobject"][obj_ind]
        dx = objlist["dx"][obj_ind]
        dy = objlist["dy"][obj_ind]
        # Make obj
        gal = gal.shear(g1=g1, g2=g2)
        obj = galsim.Convolve([gal, psf])

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
        stamp_offset = galsim.PositionD(
            nominal_x - stamp_center.x, nominal_y - stamp_center.y
        )

        if draw_method == "phot":
            stamp_size = 150
            rng_draw = rng_galsim
            maxN = int(1e6)
        else:
            stamp_size = obj.getGoodImageSize(roman.pixel_scale)
            rng_draw = None
            maxN = None

        bounds = galsim.BoundsI(1, stamp_size, 1, stamp_size)
        bounds = bounds.shift(stamp_center - bounds.center)
        stamp_image = galsim.Image(bounds=bounds, wcs=wcs)

        obj.drawImage(
            bandpass=bp_,
            image=stamp_image,
            wcs=wcs.local(world_pos=world_pos),
            method=draw_method,
            offset=stamp_offset,
            rng=rng_draw,
            maxN=maxN,
            add_to_image=True,
        )

        b = stamp_image.bounds & final_img.bounds
        if b.isDefined():
            final_img[b] += stamp_image[b]

        final_img /= roman.gain
        final_img += noise_img
        epoch_dict["sci"] = final_img.array
        epoch_dict["noise"] = noise_img.array
        epoch_dict["noise_var"] = noise_img.array.var()
        epoch_dict["weight"] = (
            np.ones_like(noise_img.array) / noise_img.array.var()
        )
    return epoch_dict


def get_deconv_psf(
    psf,
    wcs,
    cell_center_world,
    bp=None,
    oversamp_factor=3,
    chromatic=False,
    avg_gal_sed_path=None,
):
    star_psf = galsim.DeltaFunction()

    # Average galaxy SED
    if chromatic:
        sed_wave, avg_gal_sed_arr = np.load(avg_gal_sed_path)
        sed_lt = galsim.LookupTable(
            sed_wave, avg_gal_sed_arr, interpolant="linear"
        )
        avg_gal_sed = galsim.SED(sed_lt, wave_type="nm", flux_type="fnu")
        star_psf *= avg_gal_sed

    wcs_oversampled = make_oversample_local_wcs(
        wcs,
        cell_center_world,
        oversamp_factor,
    )
    wcs_local_center = wcs.local(world_pos=cell_center_world)
    pixel_ori = wcs_local_center.toWorld(galsim.Pixel(1))
    psf_obj = galsim.Convolve(psf, star_psf)
    psf_obj = galsim.Convolve(psf_obj, pixel_ori)
    psf_img = psf_obj.drawImage(
        nx=151,
        ny=151,
        wcs=wcs_oversampled,
        bandpass=bp,
        method="no_pixel",
    )

    return psf_img.array, wcs_oversampled
