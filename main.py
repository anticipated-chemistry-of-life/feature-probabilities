# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
df = pd.read_parquet("cfmid_scores.parquet")
df["is_correct"] = (df["inchikey_isdb"] == df["inchikey_msg"]).astype(int)
# Recover precursor m/z from the two difference columns (ppm = diff/mz * 1e6).
# Guard against the zero-ppm edge case to avoid inf/NaN.
df["precursor_mz"] = df["abs_precursor_mz_diff"] / df["ppm_precursor_mz_diff"] * 1e6

# drop all rows where entropy_similarity and ModifiedCosineGreedy are 0.0
df = df[(df["entropy_similarity"] != 0.0)]

# %%
# we plot the density of entropy_similarity split by instrument type (instrument column) and is_correct
plt.figure(figsize=(10, 5))
plt.hist(
    df[df["is_correct"] == 1]["entropy_similarity"],
    bins=500,
    alpha=0.5,
    label="correct",
    density=True,
)
plt.hist(
    df[df["is_correct"] == 0]["entropy_similarity"],
    bins=500,
    alpha=0.5,
    label="incorrect",
    density=True,
)
plt.legend()
plt.xlabel("entropy_similarity")
plt.ylabel("Density")
plt.title("Distribution of entropy_similarity by is_correct")
plt.show()
