from numpy import array
import sunpy.map
import dask.array
import astropy.wcs
import astropy.units as u
import reproject

m = sunpy.map.Map("/Users/willbarnes/sunpy/data/aia.lev1_euv_12s.2012-09-24T145612Z.171.image_lev1.fits")

m = m.resample([500, 500]*u.pix)

da_data = dask.array.from_array(m.data, chunks=m.data.shape)#(1000, 1000))

m_dask = sunpy.map.Map(da_data, m.meta)

target_wcs = astropy.wcs.WCS({
    "CTYPE1": "HPLN-TAN",
    "CTYPE2": "HPLT-TAN",
    "CRVAL1": 0,
    "CRVAL2": 0,
    "CRPIX1": (m.data.shape[1] + 1) / 2,
    "CRPIX2": (m.data.shape[0] + 1) / 2,
    "CDELT1": m.scale.axis1.to('arcsec / pix').value,
    "CDELT2": m.scale.axis2.to('arcsec / pix').value,
    "CUNIT1": "arcsec",
    "CUNIT2": "arcsec",
    "NAXIS1": m.data.shape[1],
    "NAXIS2": m.data.shape[0],
    "HGLN_OBS": m.meta['HGLN_OBS'],
    "HGLT_OBS": m.meta['HGLT_OBS'],
    "DSUN_OBS": m.meta["DSUN_OBS"],
    "RSUN_REF": m.meta["RSUN_REF"],
    "DATE-OBS": m.meta["DATE-OBS"],
})

#m_dask_repro = m_dask.reproject_to(target_wcs)


array_out = reproject.reproject_interp(m_dask, target_wcs, shape_out=m_dask.data.shape, return_footprint=False)

import matplotlib.pyplot as plt

plt.imshow(array_out)
plt.show()
