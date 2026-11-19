import healpy as hp
import numpy as np
from astropy.cosmology import Planck15
from astropy.table import Table, vstack
from clevar import Footprint
from clevar.cosmology import AstroPyCosmology
from scipy import stats


def main():
    mass_bins = [12.2, 12.5, 12.8, 13.1, 13.4, 13.7, 14.7]

    paths = {
        "sources": "/sps/lsst/users/lbaumont/data/HSC/hsc_y3_footprint_trim.fits",
        "group": "/sps/lsst/users/lbaumont/data/HSC/cluster_cat_newcenters.fits",
        "galaxy_random": "/sps/lsst/users/lbaumont/data/HSC/Randomsv02_trim.fits",
        "output": "/sps/lsst/users/lbaumont/data/HSC/Randoms_mass_bins.fits",
       }

    cosmo = AstroPyCosmology(be_cosmo=Planck15)

    # set number of realizations for random
    n_real = 32

    # read in some catalogs
    groups = Table.read(paths["group"])
    galaxy_random = Table.read(paths["galaxy_random"])

    # define output name
    output_name = paths["output"]

    # add mass proxy to group catalog
    alpha = 0.94
    beta = -0.24
    lum_r = np.log10(np.array(groups["LumBfunc"]))
    groups["mass_lum"] = ((lum_r - 12) - beta) / alpha + 14.0 - 2.0 * np.log10(0.7)

    # footprint
    gals = Table.read(paths["sources"])
    gals = gals[gals["hsc_y3_zbin"] > 0]
    nside = 1024 * 4
    ftpt = np.zeros(hp.nside2npix(nside))
    good = (gals["i_hsmshaperegauss_derived_weight"] > 0) & (gals["hsc_y3_zbin"] > 0)
    ftpt[
        hp.ang2pix(nside, gals["i_ra"][good], gals["i_dec"][good], lonlat=True, nest=True)
       ] = 1

    clevar_ftpt = Footprint(nside=nside, pixel=np.where(ftpt > 0)[0], nest=True    )

    # actually make the random
    create_random(n_real, galaxy_random, groups, mass_bins, cosmo, clevar_ftpt, output_name)

def create_random(
    n_real, galaxy_random, groups, mass_bins, cosmo, clevar_ftpt, output_name
):
    random_list_per_bin = []
    groups["mass_bin"] = np.digitize(groups["mass_lum"], mass_bins) - 1
    for i_bin in range(len(mass_bins) - 1):
        bin_mask = groups["mass_bin"] == i_bin
        transformed_source = quantile_normalize(
            galaxy_random["Z"], groups["Z"][bin_mask]
        )
        n_points = n_real * len(groups[bin_mask])
        rng = np.random.default_rng()

        samples = rng.choice(
            np.arange(len(galaxy_random)), size=n_points, replace=False
        )
        random_cut = galaxy_random[samples]

        random_cut.add_column(transformed_source[samples], name="Z_trans")
        random_cut.add_column(np.zeros(len(random_cut), dtype=float), name="cf_1mpc")
        random_cut.add_column(
            np.ones(len(random_cut), dtype=int) * i_bin, name="mass_bin"
        )

        # structured to allow you to easily run in chunks because it can be slow
        for i in np.arange(n_real):

            id = i * (len(groups[bin_mask])) + np.arange(len(groups[bin_mask]))
            print(id)
            random_cut["cf_1mpc"][id] = [
                clevar_ftpt.get_coverfrac(*radecz, 1, "Mpc", cosmo)
                for radecz in zip(
                    random_cut["RA"][id].value,
                    random_cut["DEC"][id].value,
                    random_cut["Z_trans"][id].value,
                )
            ]
        random_list_per_bin.append(random_cut)
    total_table = vstack(random_list_per_bin)
    total_table.write(output_name, overwrite=True)

def quantile_normalize(source_data, target_data):
    """
    Apply quantile normalization to shape the source distribution to match
    the empirical target distribution. Source and target can have different lengths.

    Parameters:
    -----------
    source_data : array-like
        Source distribution to be normalized
    target_data : array-like
        Target empirical distribution to match

    Returns:
    --------
    normalized_source : ndarray
        Source data transformed to match target distribution's quantiles
    """
    source = np.asarray(source_data).flatten()
    target = np.asarray(target_data).flatten()

    # Sort
    target_sorted = np.sort(target)

    # Compute empirical quantile ranks for source (in [0,1])
    source_ranks = (stats.rankdata(source, method="average") - 0.5) / len(source)

    # Interpolate source quantiles onto target distribution
    normalized_source = np.interp(
        source_ranks, (np.arange(len(target)) + 0.5) / len(target), target_sorted
    )

    return normalized_source

if __name__ == "__main__":
    main()
