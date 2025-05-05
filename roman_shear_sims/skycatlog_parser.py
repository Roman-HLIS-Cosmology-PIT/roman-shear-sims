import yaml
import re
import os
import h5py

import numpy as np
import pandas as pd

import galsim

from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM

from regions import PolygonSkyRegion

import healpy as hp

from .wcs import make_simple_coadd_wcs
from .utils import get_new_df_index, get_dl


HEALPIX_TEMPLATE = r"(?P<healpix>\d+)"
COMPONENTS = ["bulge", "disk", "knots"]


class SkyCatalogParser:
    def __init__(
        self,
        skycatalog_config_path,
        img_world_center,
        img_size,
        buffer=0,
        object_types=["diffsky_galaxy"],
        read_sed=True,
    ):
        self.img_world_center = img_world_center
        self.img_size = img_size
        self.buffer = buffer
        self.object_types = object_types
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
        with open(skycatalog_config_path, "r") as f:
            self.config = yaml.safe_load(f)

        object_type_config = self.config["object_types"][self.object_types[0]]
        self.hp_config = object_type_config["area_partition"]

    def _get_region(self, img_world_center, img_size, buffer=0):
        extra_buff = int(img_size * 0.1)
        extra_buff = min(extra_buff, 100)
        tmp_img_size = img_size + extra_buff

        # Get temporary WCS to query the catalog
        coadd_wcs = make_simple_coadd_wcs(
            img_world_center,
            tmp_img_size,
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
        coadd_wcs = make_simple_coadd_wcs(
            self.img_world_center,
            self.img_size - 2 * self.buffer,
            as_astropy=True,
        )

        coord = SkyCoord(
            ra=ra.to_numpy() * u.deg,
            dec=dec.to_numpy() * u.deg,
            frame="icrs",
        )
        return coadd_wcs.footprint_contains(coord)

    def _get_hp_pix_list(self):
        nside = self.hp_config["nside"]
        self.hp_pixels = hp.query_polygon(
            nside, self._reg_vert, inclusive=True
        )

    def _get_cat_paths(self, object_type, get_sed=False):
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
        return FlatLambdaCDM(
            H0=self.config["Cosmology"]["H0"],
            Om0=self.config["Cosmology"]["Om0"],
            Ob0=self.config["Cosmology"]["Ob0"],
        )

    def set_catalog(self, object_type):
        if object_type not in self.object_types:
            raise ValueError(
                f"Object type {object_type} not in {self.object_types}"
            )

        file_paths = self._get_cat_paths(object_type)
        filters = [
            ("ra", ">", self._reg_radec.vertices.ra.deg.min()),
            ("ra", "<", self._reg_radec.vertices.ra.deg.max()),
            ("dec", ">", self._reg_radec.vertices.dec.deg.min()),
            ("dec", "<", self._reg_radec.vertices.dec.deg.max()),
        ]

        cat = pd.read_parquet(
            [file_path for _, file_path in file_paths.items()],
            filters=filters,
        )

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
        if self.catalog[object_type] is None:
            self.set_catalog(object_type)
        cat = self.catalog[object_type]
        hp_pixels = cat.index.get_level_values("pixel").to_numpy()

        file_paths = self._get_cat_paths(object_type, get_sed=True)

        cosmo = self._get_cosmology()

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
                        # print(row.Index)
                        gal_ind = row.galaxy_id
                        sed_array = f_grp[str(gal_ind)][:].astype(np.float64)
                        sed_array /= (
                            4.0
                            * np.pi
                            * get_dl(cosmo, row.redshiftHubble) ** 2
                        )
                        for i, component in enumerate(COMPONENTS):
                            lut = galsim.LookupTable(
                                x=wave_list,
                                f=sed_array[i, :],
                                interpolant="linear",
                            )
                            sed = galsim.SED(
                                lut,
                                wave_type="angstrom",
                                flux_type="fnu",
                            ).atRedshift(row.redshift)
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
    Return the angular knot size. Knots are modelled as the same
    physical size
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
