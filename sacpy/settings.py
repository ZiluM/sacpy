import cartopy.crs as ccrs


class Map():
    """
    Map settings
    """
    projection_dict = {
        'Robinson': ccrs.Robinson,
        'NorthPolarStereo': ccrs.NorthPolarStereo,
        'SouthPolarStereo': ccrs.SouthPolarStereo,
        'PlateCarree': ccrs.PlateCarree,
        'AlbersEqualArea': ccrs.AlbersEqualArea,
        'AzimuthalEquidistant': ccrs.AzimuthalEquidistant,
        'EquidistantConic': ccrs.EquidistantConic,
        'LambertConformal': ccrs.LambertConformal,
        'LambertCylindrical': ccrs.LambertCylindrical,
        'Mercator': ccrs.Mercator,
        'Miller': ccrs.Miller,
        'Mollweide': ccrs.Mollweide,
        'Orthographic': ccrs.Orthographic,
        'Sinusoidal': ccrs.Sinusoidal,
        'Stereographic': ccrs.Stereographic,
        'TransverseMercator': ccrs.TransverseMercator,
        'UTM': ccrs.UTM,
        'InterruptedGoodeHomolosine': ccrs.InterruptedGoodeHomolosine,
        'RotatedPole': ccrs.RotatedPole,
        'OSGB': ccrs.OSGB,
        'EuroPP': ccrs.EuroPP,
        'Geostationary': ccrs.Geostationary,
        'NearsidePerspective': ccrs.NearsidePerspective,
        'EckertI': ccrs.EckertI,
        'EckertII': ccrs.EckertII,
        'EckertIII': ccrs.EckertIII,
        'EckertIV': ccrs.EckertIV,
        'EckertV': ccrs.EckertV,
        'EckertVI': ccrs.EckertVI,
        'EqualEarth': ccrs.EqualEarth,
        'Gnomonic': ccrs.Gnomonic,
        'LambertAzimuthalEqualArea': ccrs.LambertAzimuthalEqualArea,
        'OSNI': ccrs.OSNI,
    }
