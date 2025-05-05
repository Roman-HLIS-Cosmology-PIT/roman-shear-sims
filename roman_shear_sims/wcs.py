import numpy as np

import galsim
from galsim import roman

from astropy.io.fits.header import Header
from astropy.wcs import WCS

import coord


def get_SCA_WCS(world_pos, SCA, PA=0.0, img_size=None):
    a_sip, b_sip = roman.roman_wcs._parse_sip_file(
        roman.roman_wcs.sip_filename
    )
    n_sip = roman.roman_wcs.n_sip

    PA *= galsim.degrees
    crval = world_pos
    i_sca = SCA

    if img_size is None:
        img_size = roman.n_pix

    # Leave phi_p at 180 (0 if dec_targ==-90), so that tangent plane axes remain oriented along
    # celestial coordinates. In other words, phi_p is the angle of the +Y axis in the tangent
    # plane, which is of course pi if we're measuring these phi angles clockwise from the -Y
    # axis.  Note that this quantity is not used in any calculations at all, but for consistency
    # with the WCS code that comes from the Roman project office, we calculate this quantity
    # and put it in the FITS header.
    if world_pos.dec / coord.degrees > -90.0:
        phi_p = np.pi * coord.radians
    else:
        phi_p = 0.0 * coord.radians

    # Compute the position angle of the local pixel Y axis.
    # This requires projecting local North onto the detector axes.
    # Start by adding any SCA-unique rotation relative to FPA axes:
    sca_tp_rot = PA

    # Go some reasonable distance from crval in the +y direction.  Say, 1 degree.
    u, v = world_pos.project(crval, projection="gnomonic")
    plus_y = world_pos.deproject(
        u, v + 1 * coord.degrees, projection="gnomonic"
    )
    # Find the angle between this point, crval and due north.
    north = coord.CelestialCoord(0.0 * coord.degrees, 90.0 * coord.degrees)
    pa_sca = sca_tp_rot - crval.angleBetween(plus_y, north)

    # Compute CD coefficients: extract the linear terms from the a_sip, b_sip arrays.  These
    # linear terms are stored in the SIP arrays for convenience.
    a10 = a_sip[i_sca, 1, 0]
    a11 = a_sip[i_sca, 0, 1]
    b10 = b_sip[i_sca, 1, 0]
    b11 = b_sip[i_sca, 0, 1]

    # Rotate by pa_fpa.
    cos_pa_sca = np.cos(pa_sca)
    sin_pa_sca = np.sin(pa_sca)

    header = []
    roman.roman_wcs._populate_required_fields(header)
    header.extend(
        [
            (
                "CRVAL1",
                crval.ra / coord.degrees,
                "first axis value at reference pixel",
            ),
            (
                "CRVAL2",
                crval.dec / coord.degrees,
                "second axis value at reference pixel",
            ),
            (
                "CD1_1",
                cos_pa_sca * a10 + sin_pa_sca * b10,
                "partial of first axis coordinate w.r.t. x",
            ),
            (
                "CD1_2",
                cos_pa_sca * a11 + sin_pa_sca * b11,
                "partial of first axis coordinate w.r.t. y",
            ),
            (
                "CD2_1",
                -sin_pa_sca * a10 + cos_pa_sca * b10,
                "partial of second axis coordinate w.r.t. x",
            ),
            (
                "CD2_2",
                -sin_pa_sca * a11 + cos_pa_sca * b11,
                "partial of second axis coordinate w.r.t. y",
            ),
            (
                "ORIENTAT",
                pa_sca / coord.degrees,
                "position angle of image y axis (deg. e of n)",
            ),
            (
                "LONPOLE",
                phi_p / coord.degrees,
                "Native longitude of celestial pole",
            ),
        ]
    )
    for i in range(n_sip):
        for j in range(n_sip):
            if i + j >= 2 and i + j < n_sip:
                sipstr = "A_%d_%d" % (i, j)
                header.append((sipstr, a_sip[i_sca, i, j]))
                sipstr = "B_%d_%d" % (i, j)
                header.append((sipstr, b_sip[i_sca, i, j]))

    header = galsim.FitsHeader(header)
    if img_size is not None:
        header["CRPIX1"] = img_size / 2
        header["CRPIX2"] = img_size / 2
    wcs = galsim.GSFitsWCS(header=header)
    wcs.header = header

    return wcs


def make_simple_exp_wcs(world_pos, PA=0.0, img_size=None):
    PA *= galsim.degrees
    crval = world_pos

    if img_size is None:
        img_size = roman.n_pix

    image_center = galsim.PositionD(img_size / 2, img_size / 2)

    # Compute the position angle of the local pixel Y axis.
    # This requires projecting local North onto the detector axes.
    # Start by adding any SCA-unique rotation relative to FPA axes:
    sca_tp_rot = PA

    # Go some reasonable distance from crval in the +y direction.  Say, 1 degree.
    u, v = world_pos.project(crval, projection="gnomonic")
    plus_y = world_pos.deproject(
        u, v + 1 * coord.degrees, projection="gnomonic"
    )
    # Find the angle between this point, crval and due north.
    north = coord.CelestialCoord(0.0 * coord.degrees, 90.0 * coord.degrees)
    pa_sca = sca_tp_rot - crval.angleBetween(plus_y, north)

    # Rotate by pa_fpa.
    cos_pa_sca = np.cos(pa_sca)
    sin_pa_sca = np.sin(pa_sca)

    pixel_scale = roman.pixel_scale

    dudx = cos_pa_sca * pixel_scale
    dudy = -sin_pa_sca * pixel_scale
    dvdx = sin_pa_sca * pixel_scale
    dvdy = cos_pa_sca * pixel_scale

    affine = galsim.AffineTransform(
        dudx, dudy, dvdx, dvdy, origin=image_center
    )
    wcs = galsim.TanWCS(affine, world_pos, units=galsim.arcsec)
    return wcs


def make_simple_coadd_wcs(world_pos, img_size, as_astropy=False):
    pixel_scale = roman.pixel_scale

    image_origin = galsim.PositionD(img_size / 2, img_size / 2)

    mat = np.array([[pixel_scale, 0], [0, pixel_scale]])

    affine = galsim.AffineTransform(
        mat[0, 0], mat[0, 1], mat[1, 0], mat[1, 1], origin=image_origin
    )
    wcs = galsim.TanWCS(affine, world_pos, units=galsim.arcsec)

    if not as_astropy:
        return wcs
    else:
        h = Header()
        b = galsim.BoundsI(1, img_size, 1, img_size)
        wcs.writeToFitsHeader(h, b)
        h["NAXIS"] = 2
        h["NAXIS1"] = img_size
        h["NAXIS2"] = img_size
        return WCS(h)


def make_oversample_local_wcs(wcs, world_pos, oversamp_factor):
    local_wcs = wcs.local(world_pos=world_pos)
    new_jac_matrix = local_wcs.getMatrix() / oversamp_factor
    wcs_oversampled = galsim.JacobianWCS(*new_jac_matrix.ravel())

    return wcs_oversampled
