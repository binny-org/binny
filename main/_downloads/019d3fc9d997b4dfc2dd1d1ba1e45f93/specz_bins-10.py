import numpy as np

from binny import NZTomography

tomo = NZTomography()

z = np.linspace(0.0, 1.0, 500)

nz = NZTomography.nz_model(
    "smail",
    z,
    z0=0.12,
    alpha=2.0,
    beta=1.5,
    normalize=True,
)

specz_spec = {
    "kind": "specz",
    "bins": {
        "scheme": "equipopulated",
        "n_bins": 4,
    },
    "uncertainties": {
        "completeness": [1.0, 0.95, 0.9, 0.85],
        "catastrophic_frac": [0.0, 0.05, 0.1, 0.15],
        "leakage_model": "neighbor",
        "specz_scatter": [0.001, 0.001, 0.0015, 0.002],
    },
    "normalize_bins": True,
}

specz_result = tomo.build_bins(z=z, nz=nz, tomo_spec=specz_spec)

print("Bin keys:")
print(list(specz_result.bins.keys()))
print()
print("Parent shape:", specz_result.nz.shape)
first_key = sorted(specz_result.bins.keys())[0]
print("First bin shape:", specz_result.bins[first_key].shape)
print("Resolved kind:", specz_result.spec["kind"])
print("Resolved scheme:", specz_result.spec["bins"]["scheme"])
print("Resolved n_bins:", specz_result.spec["bins"]["n_bins"])