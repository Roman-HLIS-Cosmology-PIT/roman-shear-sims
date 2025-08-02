import yaml
import os
import h5py

import numpy as np
import pandas as pd
import polars as pl

import galsim
import galsim.roman as roman

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM

from regions import PolygonSkyRegion

import healpy as hp

from .wcs import make_simple_coadd_wcs, get_IMCOM_WCS
from .utils import get_new_df_index, get_dl


HEALPIX_TEMPLATE = r"(?P<healpix>\d+)"
COMPONENTS = ["bulge", "disk", "knots"]


class SkyCatalogParser:
    """
    Parse the OpenUniverse2024 SkyCatalog.

    Parameters
    ----------
    skycatalog_config_path : str
        The path to the SyCatalog .yaml configuration file.
    img_world_center : galsim.CelestialCoord
        The celestial coordinates of the image center.
    img_size : int
        The size of the image in pixels.
    simu_type : str, optional
        The type of simulation to run. Options are 'sca' for SCA simulation
        and 'imcom' for IMCOM simulation. Default: 'sca'.
    buffer : int, optional
        The buffer size in pixels to extend the image region for
        catalog queries.
    object_types : list of str, optional
        The list of object types to parse from the catalog.
        Default is ['diffsky_galaxy'].
    read_sed : bool, optional
        Whether to read the SEDs from the catalog.
        Default is True.
    """

    def __init__(
        self,
        skycatalog_config_path,
        img_world_center,
        img_size,
        simu_type="sca",
        buffer=0,
        object_types=None,
        read_sed=True,
    ):
        self.img_world_center = img_world_center
        self.img_size = img_size
        self._simu_type = simu_type
        self.buffer = buffer
        if object_types is None:
            self._object_types = ["diffsky_galaxy"]
        else:
            self._object_types = object_types
        self._read_sed = read_sed

        self._init_catalog()
        self._parse_skycatalog(skycatalog_config_path)
        self._get_region(img_world_center, img_size, buffer)
        self._get_hp_pix_list()

    def _init_catalog(self):
        self.catalog = {object_type: None for object_type in self.object_types}
        if self._read_sed:
            self.sed_catalog = {
                object_type: None for object_type in self.object_types
            }

    def _parse_skycatalog(self, skycatalog_config_path):
        """
        Parse the SkyCatalog configuration file.
        """
        with open(skycatalog_config_path, "r") as f:
            self.config = yaml.safe_load(f)

        object_type_config = self.config["object_types"][self.object_types[0]]
        self.hp_config = object_type_config["area_partition"]

    def _get_region(self, img_world_center, img_size, buffer=0):
        """
        Get the region to query the catalog based on the image center and size.
        This reguion is used for the initial query of the catalog using:
        min_ra < ra < max_ra and min_dec < dec < max_dec
        The region is extended by a buffer to ensure that the entire image is
        covered by the catalog query.
        """
        extra_buff = int(img_size * 0.1)
        extra_buff = min(extra_buff, 100)
        tmp_img_size = img_size + extra_buff

        # Get temporary WCS to query the catalog
        if self._simu_type == "sca":
            coadd_wcs = make_simple_coadd_wcs(
                img_world_center,
                tmp_img_size,
                as_astropy=True,
            )
        elif self._simu_type == "imcom":
            coadd_wcs = get_IMCOM_WCS(
                img_world_center,
                img_size=tmp_img_size,
                as_astropy=True,
            )

        # Image footprint region
        coord_corners = SkyCoord(
            ra=coadd_wcs.calc_footprint()[:, 0] * u.deg,
            dec=coadd_wcs.calc_footprint()[:, 1] * u.deg,
            frame="icrs",
        )
        self._reg_radec = PolygonSkyRegion(coord_corners)
        reg_vert = hp.ang2vec(
            self._reg_radec.vertices.ra.deg,
            self._reg_radec.vertices.dec.deg,
            lonlat=True,
        )
        self._reg_vert = reg_vert

    def _get_in_img_footprint(self, ra, dec):
        """
        Check if the given RA and Dec are within the image footprint.
        Compared to the _get_region method, this method uses the image WCS to
        check each object is within the image footprint + buffer.
        """
        if self._simu_type == "sca":
            coadd_wcs = make_simple_coadd_wcs(
                self.img_world_center,
                self.img_size - 2 * self.buffer,
                as_astropy=True,
            )
        elif self._simu_type == "imcom":
            coadd_wcs = get_IMCOM_WCS(
                self.img_world_center,
                img_size=self.img_size - 2 * self.buffer,
                as_astropy=True,
            )

        coord = SkyCoord(
            ra=ra.to_numpy() * u.deg,
            dec=dec.to_numpy() * u.deg,
            frame="icrs",
        )
        return coadd_wcs.footprint_contains(coord)

    def _get_hp_pix_list(self):
        """
        Get the HEALPix pixel list that overlaps with the image region.
        """
        nside = self.hp_config["nside"]
        self.hp_pixels = hp.query_polygon(
            nside, self._reg_vert, inclusive=True
        )

    def _get_cat_paths(self, object_type, get_sed=False):
        """
        Get the catalog file paths for a given object type.
        """
        root_dir = self.config["catalog_dir"]
        if not get_sed:
            cat_template = self.config["object_types"][object_type][
                "file_template"
            ]
        else:
            cat_template = self.config["object_types"][object_type][
                "sed_file_template"
            ]
        file_paths = {}
        for pixel in self.hp_pixels:
            cat_name = cat_template.replace(HEALPIX_TEMPLATE, str(pixel))
            cat_path = os.path.join(root_dir, cat_name)
            if not os.path.exists(cat_path):
                # raise FileNotFoundError(
                #     f"Catalog file {cat_path} does not exist."
                # )
                continue
            file_paths[pixel] = cat_path
        return file_paths

    def _get_cosmology(self):
        """
        Get the cosmology used to create the OpenUniverse2024.
        """
        return FlatLambdaCDM(
            H0=self.config["Cosmology"]["H0"],
            Om0=self.config["Cosmology"]["Om0"],
            Ob0=self.config["Cosmology"]["Ob0"],
        )

    def set_catalog(self, object_type):
        """
        Set the catalog for a given object type.
        This method queries the catalog files for the given object type and
        filters the objects based on the image region.

        NOTE: Probably only works for the diffsky_galaxy object type at the
        moment.

        Parameters
        ----------
        object_type : str
            The type of object to set the catalog for.
        """
        if object_type not in self.object_types:
            raise ValueError(
                f"Object type {object_type} not in {self.object_types}"
            )

        file_paths = self._get_cat_paths(object_type)

        q = (
            pl.scan_parquet(
                [file_path for _, file_path in file_paths.items()],
            )
            .drop(
                "peculiarVelocity",
                "shear1",
                "shear2",
                "convergence",
                "MW_rv",
                "MW_av",
            )
            .filter(
                (pl.col("ra") > self._reg_radec.vertices.ra.deg.min())
                & (pl.col("ra") < self._reg_radec.vertices.ra.deg.max())
                & (pl.col("dec") > self._reg_radec.vertices.dec.deg.min())
                & (pl.col("dec") < self._reg_radec.vertices.dec.deg.max())
            )
        )
        cat = q.collect().to_pandas()

        final_mask = self._get_in_img_footprint(cat["ra"], cat["dec"])
        cat = cat[final_mask]
        if object_type == "diffsky_galaxy":
            new_ind = get_new_df_index(cat["galaxy_id"].to_numpy())
            cat = cat.set_index(new_ind).sort_index()
            self.catalog[object_type] = cat.copy()
        elif object_type == "star":
            self.catalog[object_type] = cat.copy()

        del cat
        del final_mask

    def set_sed_catalog(self, object_type):
        """
        Set the SED catalog for a given object type.
        This method reads the SEDs from the catalog files for the given object
        type and stores them in the sed_catalog attribute.

        NOTE: Probably only works for the diffsky_galaxy object type at the
        moment.

        Parameters
        ----------
        object_type : str
            The type of object to set the catalog for.
        """
        if self.catalog[object_type] is None:
            self.set_catalog(object_type)
        cat = self.catalog[object_type]
        hp_pixels = cat.index.get_level_values("pixel").to_numpy()

        file_paths = self._get_cat_paths(object_type, get_sed=True)

        cosmo = self._get_cosmology()

        # Get lower/higher limirs of the SED
        blue_lim = roman.getBandpasses()["Y106"].blue_limit * 10
        red_lim = roman.getBandpasses()["H158"].red_limit * 10

        seds = {key: [] for key in COMPONENTS}
        inds = []
        for pixel in np.unique(hp_pixels):
            with h5py.File(file_paths[pixel], "r") as f:
                wave_list = f["meta"]["wave_list"][:]
                sub_cat = cat.loc[pixel]
                for sed_ind in np.unique(
                    sub_cat.index.get_level_values("sed_ind").to_numpy()
                ):
                    f_grp = f[f"galaxy/{sed_ind}"]
                    for row in sub_cat.loc[int(sed_ind)].itertuples():
                        gal_ind = row.galaxy_id
                        start, end, z_wave_list = get_redshift_ind(
                            wave_list,
                            row.redshift,
                            blue_lim,
                            red_lim,
                        )
                        sed_array = f_grp[str(gal_ind)][:, start:end].astype(
                            np.float64
                        )
                        sed_array /= (
                            4.0
                            * np.pi
                            * get_dl(cosmo, row.redshiftHubble) ** 2
                        )
                        for i, component in enumerate(COMPONENTS):
                            lut = galsim.LookupTable(
                                # x=wave_list,
                                x=z_wave_list,
                                f=sed_array[i, :] * (1 + row.redshift),
                                interpolant="linear",
                            )
                            sed = galsim.SED(
                                lut,
                                wave_type="angstrom",
                                flux_type="fnu",
                            )
                            seds[component].append(sed)
                        inds.append(row.Index)
        self.sed_catalog[object_type] = pd.DataFrame(
            seds,
            index=np.array(inds),
        ).sort_index()

    # def get_catalog(self, object_type):
    #     if self._catalog[object_type] is None:
    #         self._set_catalog(object_type)
    #     return self._catalog[object_type]


def get_knot_size(z):
    """
    Return the angular knot size. Knots are modelled as the same physical size.

    Parameters
    ----------
    z : float
        The redshift of the galaxy.

    Returns
    -------
    float or None
        The angular size of the knots in arcseconds, or None if the redshift is
        above 0.6 (where knots are treated as point sources).
    """
    # Deceleration paramameter
    q = -0.5

    if z >= 0.6:
        # Above z=0.6, fractional contribution to post-convolved size
        # is <20% for smallest Roman PSF size, so can treat as point source
        # This also ensures sqrt in formula below has a
        # non-negative argument
        return None

    # Angular diameter scaling approximation in pc
    dA = (
        (3e9 / q**2)
        * (z * q + (q - 1) * (np.sqrt(2 * q * z + 1) - 1))
        / (1 + z) ** 2
        * (1.4 - 0.53 * z)
    )
    # Using typical knot size 250pc, convert to sigma in arcmin
    return 206264.8 * 250 / dA / 2.355


def get_knot_n(um_source_galaxy_obs_sm, gal_id=None, rng=None):
    """
    Return random value for number of knots based on galaxy sm.

    Parameters
    ----------
    um_source_galaxy_obs_sm : float
        The observed stellar mass of the galaxy.
    gal_id : int, optional
        The galaxy ID to use for the random number generator.
    rng : galsim.BaseDeviate, optional
        The random number generator to use. If None, a new one is created using
        the gal_id.

    Returns
    -------
    int
        The number of knots for the galaxy.
    """
    if rng is not None:
        ud = galsim.UniformDeviate(rng)
    else:
        if gal_id is None:
            raise ValueError("Either rng or gal_id must be provided.")
        ud = galsim.UniformDeviate(int(gal_id))
    sm = np.log10(um_source_galaxy_obs_sm)
    m = (50 - 3) / (12 - 6)  # (knot_n range)/(logsm range)
    n_knot_max = m * (sm - 6) + 3
    n_knot = int(ud() * n_knot_max)  # random n up to n_knot_max
    if n_knot == 0:
        n_knot += 1  # need at least 1 knot
    return n_knot


def get_redshift_ind(wave_list, redshift, blue_limit, red_limit):
    """
    This is used to only load the part of the SED that is relevant for the
    given redshift and wavelength range covered by the bandpasses.

    Parameters
    ----------
    wave_list : np.ndarray
        The original wavelength list of the SED.
    redshift : float
        The redshift of the galaxy.
    blue_limit : float
        The blue limit of the wavelength range.
    red_limit : float
        The red limit of the wavelength range.

    Returns
    -------
    tuple
        start : int
            The starting index of the wavelength range.
        end : int
            The ending index of the wavelength range.
        z_wave_list : np.ndarray
            The wavelength list after applying the redshift.
    """
    z_factor = 1 + redshift
    z_wave_list = wave_list * z_factor
    good_ind = np.where(
        (z_wave_list >= blue_limit) & (z_wave_list <= red_limit)
    )[0]
    start = good_ind[0]
    end = good_ind[-1]
    if start - 1 >= 0:
        start -= 1
    if end + 1 < len(z_wave_list):
        end += 1
    end += 1

    return start, end, z_wave_list[start:end]
