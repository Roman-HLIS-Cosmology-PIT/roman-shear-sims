import numpy as np

import galsim
from galsim import roman

from roman_imsim import SkyCatalogInterface


class SkyCatalogInterfaceNew(SkyCatalogInterface):
    def get_gsobject_components(self, cls, rng, gsparams=None):
        if gsparams is not None:
            gsparams = galsim.GSParams(**gsparams)

        obj_dict = {}
        for component in cls.subcomponents:
            # knots use the same major/minor axes as the disk component.
            my_cmp = "disk" if component != "bulge" else "spheroid"
            hlr = cls.get_native_attribute(f"{my_cmp}HalfLightRadiusArcsec")

            # Get ellipticities saved in catalog. Not sure they're what
            # we need
            e1 = cls.get_native_attribute(f"{my_cmp}Ellipticity1")
            e2 = cls.get_native_attribute(f"{my_cmp}Ellipticity2")
            shear = galsim.Shear(g1=e1, g2=e2)

            if component == "knots":
                npoints = cls.get_knot_n()
                assert npoints > 0
                knot_profile = galsim.Sersic(
                    n=cls._sersic_disk,
                    half_light_radius=hlr / 2.0,
                    gsparams=gsparams,
                )
                knot_profile = knot_profile._shear(shear)
                obj = galsim.RandomKnots(
                    npoints=npoints,
                    profile=knot_profile,
                    rng=rng,
                    gsparams=gsparams,
                )
                z = cls.get_native_attribute("redshift")
                size = cls.get_knot_size(z)  # get knot size
                if size is not None:
                    obj = galsim.Convolve(obj, galsim.Gaussian(sigma=size))
                obj_dict[component] = obj
            else:
                n = (
                    cls._sersic_disk
                    if component == "disk"
                    else cls._sersic_bulge
                )
                obj = galsim.Sersic(
                    n=n, half_light_radius=hlr, gsparams=gsparams
                )
                obj_dict[component] = obj._shear(shear)

        return obj_dict

    def getFlux(self, skycat_obj, filter=None):
        """
        Return the flux associated to an object.

        Parameters
        ----------
        index : int
            Index of the object in the self.objects catalog.
        filter : str, optional
            Name of the filter for which the flux is computed. If None, use the
            filter provided during initialization. [Default: None]

        Returns
        -------
        flux
            Computer flux at the given date for the requested exposure time and
            filter.
        """

        if filter is None:
            filter = self.bandpass.name

        # We cache the SEDs for potential later use
        self._seds = skycat_obj.get_observer_sed_components(mjd=self.mjd)
        for i, sed in enumerate(self._seds.values()):
            if i == 0:
                sed_sum = sed
            else:
                sed_sum += sed
        flux = skycat_obj.get_roman_flux(filter, sed_sum, cache=False)

        return flux

    def getObj(self, index, rng=None, gsparams=None, bandpass=None):
        if not self.objects:
            raise RuntimeError(
                "Trying to get an object from an empty sky catalog"
            )

        if bandpass is None:
            bandpass = self.bandpass

        skycat_obj = self.objects[index]

        # Use the pre-computed flux and skip if the object is too faint
        flux_quick = (
            skycat_obj.get_native_attribute(f"roman_flux_{bandpass.name}")
            * self.exptime
            * roman.collecting_area
        )
        if np.isnan(flux_quick):
            return None
        if flux_quick < 60:
            return None

        flux = (
            self.getFlux(
                skycat_obj,
                filter=bandpass.name,
            )
            * self.exptime
            * roman.collecting_area
        )
        if np.isnan(flux):
            return None
        gsobjs = self.get_gsobject_components(skycat_obj, rng, gsparams)
        if hasattr(self, "_seds"):
            seds = self._seds
        else:
            seds = skycat_obj.get_observer_sed_components(mjd=self.mjd)

        gs_obj_list = []
        for component in gsobjs:
            if component in seds:
                gs_obj_list.append(
                    gsobjs[component]
                    * seds[component]
                    * self.exptime
                    * roman.collecting_area
                )
        if not gs_obj_list:
            return None

        if len(gs_obj_list) == 1:
            gs_object = gs_obj_list[0]
        else:
            gs_object = galsim.Add(gs_obj_list)

        # Give the object the right flux
        gs_object.flux = flux
        gs_object.withFlux(gs_object.flux, bandpass)

        # Get the object type
        if (skycat_obj.object_type == "diffsky_galaxy") | (
            skycat_obj.object_type == "galaxy"
        ):
            gs_object.object_type = "galaxy"
        if skycat_obj.object_type == "star":
            gs_object.object_type = "star"
        if skycat_obj.object_type == "snana":
            gs_object.object_type = "transient"

        return gs_object
