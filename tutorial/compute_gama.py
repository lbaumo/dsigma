import argparse
import os

import numpy as np
from astropy import units as u
from astropy.cosmology import units as cu
from astropy.cosmology import Planck15
from astropy.table import Table

from dsigma.precompute import precompute
from dsigma.jackknife import compress_jackknife_fields
from dsigma.jackknife import compute_jackknife_fields, jackknife_resampling
from dsigma.stacking import excess_surface_density

parser = argparse.ArgumentParser(
    description='Calculate the lensing signal around BOSS galaxies.')

parser.add_argument('survey', help='the lensing survey')
args = parser.parse_args()

cosmology = Planck15
rp_bins = np.logspace(np.log10(0.1), np.log10(7), 14) * u.Mpc
m_bins = np.array([0.15, 0.31, 0.43, 0.54, 0.70])

#prepare lens table
table_l = Table.read('/sps/lsst/users/lbaumont/data/HSC/cluster_coverfraction.fits')
# trim to keep lenses that are at least half covered to 1 mpc
table_l=table_l[table_l["cf_1mpc"] > 0.5]
# rename to dsigma columns
lens_keys = dict(z="IterCenZ", ra="BCGRA", dec="BCGDec", mhalo="ma\
ss_lum")
for new_key, old_key in lens_keys.items():
    table_l.rename_column(old_key, new_key)
table_l.keep_columns(lens_keys.keys())
table_l["w_sys"] = 1

#prepare random table
table_r = Table.read('/sps/lsst/users/lbaumont/data/HSC/random_mass_bin_13_14.f\
its')
random_keys = dict(z="Z_trans", ra="RA", dec="DEC", mbin="mass_bin")

for new_key, old_key in random_keys.items():
    table_r.rename_column(old_key, new_key)
table_r.keep_columns(random_keys.keys())
table_r['w_sys'] = 1

# source table and calibration tables
table_s = Table.read('hsc_y3.hdf5', path='catalog')
table_n = Table.read('hsc_y3.hdf5', path='calibration')

z_ave = np.array([np.average(table_n['z'], weights=table_n['n'][:, z_bin])
                      for z_bin in range(4)])
table_s['z_l_max'] = z_ave[table_s['z_bin']] - 0.3

stacking_kwargs = dict(scalar_shear_response_correction=True,
                           shear_responsivity_correction=True,
                           selection_bias_correction=True, boost_correction=True,)


precompute_kwargs = dict(
    table_n=table_n, n_jobs=os.cpu_count(), comoving=False,
    cosmology=cosmology, progress_bar=True)

# Pre-compute the signal.
precompute(table_l, table_s, rp_bins, **precompute_kwargs)
precompute(table_r, table_s, rp_bins, **precompute_kwargs)

# Drop all lenses and randoms that did not have any nearby source.
table_l = table_l[np.sum(table_l['sum 1'], axis=1) > 0]
table_r = table_r[np.sum(table_r['sum 1'], axis=1) > 0]

centers = compute_jackknife_fields(
    table_l, 100, weights=np.sum(table_l['sum 1'], axis=1), seed=0)
compute_jackknife_fields(table_r, centers)

# Stack the signal.
stacking_kwargs['random_subtraction'] = True
stacking_kwargs["return_table"] = False

for lens_bin in range(len(m_bins) - 1):

    mask_r=(table_r["mass_bin"] == lens_bin)
    mask_l = (m_bins[lens_bin] <= table_l["mhalo"]) & (
        table_l["mhalo"] < m_bins[lens_bin + 1]
    )
    n_lenses = np.sum(mask_l)
    print(f"Bin {lens_bin}: {n_lenses} lenses processed")
    # mask_r = ((m_bins[lens_bin] <= table_r['mhalo']) &                        
    #          (table_r['mhalo'] < m_bins[lens_bin + 1]))  
    table_l_bin = compress_jackknife_fields(
        table_l[mask_l])
    table_r_bin = compress_jackknife_fields(
        table_r[mask_r])
    # perform the stacking
    result = excess_surface_density(
        table_l_bin, table_r=table_r_bin, return_table=True, **stacking_kwargs)
    
    result['ds_t_err'] = np.sqrt(np.diag(jackknife_resampling(
        excess_surface_density, table_l_bin, table_r=table_r_bin,
        **stacking_kwargs)))

    # stack cross component
    stacking_kwargs_cross = stacking_kwargs.copy()
    stacking_kwargs_cross['cross_component'] = True
    

    result['ds_x'] = excess_surface_density(table_l[mask_l], **stacking_kwargs_\
cross)
    result['ds_x_err'] = np.sqrt(np.diag(jackknife_resampling(
        excess_surface_density, table_l_bin,table_r=table_r_bin, **stacking_kwargs_cross)))

    result.write(f"boost_one_bin_{args.survey.lower()}_{lens_bin}.csv", overwri\
te=True)

    for key in result.colnames:
        # Drop little h from units.
        if key[:2] == 'rp':
            result[key] = result[key].to(u.Mpc, cu.with_H0(Planck15.H0))
        if key[:2] == 'ds':
            result[key] = result[key].to(
                u.Msun / u.pc**2, cu.with_H0(Planck15.H0))

    result.write(f'{args.survey.lower()}_{lens_bin}.ecsv', overwrite=True)
