import numpy as np
from binny import NZTomography

tomo = NZTomography()

z = np.linspace(0.0, 2.0, 500)

nz = NZTomography.nz_model(
    "smail",
    z,
    z0=0.5,
    alpha=2.0,
    beta=1.0,
    normalize=True,
)

spec = {
    "kind": "photoz",
    "bins": {
        "scheme": "equipopulated",
        "n_bins": 4,
    },
    "uncertainties": {
        "scatter_scale": [0.03, 0.04, 0.05, 0.06],
        "mean_offset": [0.00, 0.01, 0.01, 0.02],
        "mean_scale": [1.00, 1.00, 1.00, 1.00],
        "outlier_frac": [0.00, 0.02, 0.03, 0.05],
        "outlier_scatter_scale": [0.00, 0.20, 0.25, 0.30],
        "outlier_mean_offset": [0.00, 0.05, 0.05, 0.08],
        "outlier_mean_scale": [1.00, 1.00, 1.00, 1.00],
    },
    "normalize_bins": True,
}

result = tomo.build_bins(z=z, nz=nz, tomo_spec=spec)

print("bin keys:", list(result.bins.keys()))
print("parent shape:", result.nz.shape)
print("bin 0 shape:", result.bins[0].shape)
print("resolved scheme:", result.spec["bins"]["scheme"])