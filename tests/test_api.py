import numpy as np

import binny as bn


def test_public_api_smoke():
    z = np.linspace(0.0, 3.0, 101)
    nz = bn.redshift_distribution("smail", z, z0=0.5, alpha=2.0, beta=1.5)

    edges = [0.0, 0.5, 1.0, 1.5]
    bins = bn.photoz_bins(
        z=z,
        nz=nz,
        bin_edges=edges,
        scatter_scale=0.05,
        mean_offset=0.01,
    )

    assert set(bins.keys()) == {0, 1, 2}
    centers = bn.bin_centers(z, bins, method="mean", decimal_places=None)
    assert len(centers) == 3
