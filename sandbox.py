import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import matplotlib.pyplot as plt
    import polars as pl
    import seaborn as sns
    import numpy as np

    return np, pl, plt, sns


@app.cell
def _(pl):
    msg = pl.read_csv(
        "hf://datasets/roman-bushuiev/MassSpecGym/data/MassSpecGym1.5.tsv",
        separator="\t",
    )
    return (msg,)


@app.cell
def _(msg, pl):
    # `Series.map(dict)` becomes a left join. Deduplicating with keep="last"
    # reproduces `set_index(...).to_dict()`, which keeps the last value for a
    # repeated key; without it, duplicate identifiers would fan out rows.
    inchikey_map = msg.select(
        pl.col("identifier").alias("mappingFeatureId"),
        pl.col("inchikey").alias("inchikey_msg"),
        pl.col("fold"),
    ).lazy()

    df = (
        pl.scan_parquet("structure_identifications_all.parquet")
        .join(inchikey_map, on="mappingFeatureId", how="left")
        .with_columns(
            # In polars `==` propagates null, so unmatched keys would give null
            # rather than 0. pandas compares NaN as False, hence fill_null(False).
            (pl.col("InChIkey2D") == pl.col("inchikey_msg"))
            .fill_null(False)
            .cast(pl.Int8)
            .alias("is_correct")
        )
        .filter(pl.col("is_correct") == 1)
        # `isfinite | isnan` keeps everything except +/-inf. is_infinite() is
        # null for null rows, so fill_null(False) keeps missing scores as pandas did.
        .filter(~pl.col("CSI:FingerIDScore").is_infinite().fill_null(False))
        .unique(["inchikey_msg", "instrument_type"])
        .collect()
    )
    return (df,)


@app.cell
def _(df, sns):
    sns.histplot(
        data=df,
        x="CSI:FingerIDScore",
        y="ionMass",
        stat="density",
        common_norm=False,
        kde=True,
        hue="fold",
    )
    return


@app.cell
def _(df, pl):
    data = (
        df[["CSI:FingerIDScore", "ionMass"]]
        .filter(~pl.any_horizontal(pl.all().is_infinite()))
        .to_numpy()
        .T
    )
    return (data,)


@app.cell
def _(data):
    from scipy.stats import gaussian_kde
    from sklearn.preprocessing import StandardScaler

    kde = gaussian_kde(data)
    return (kde,)


@app.cell
def _(data, kde, np, plt):
    # Evaluate on a grid
    x_grid = np.linspace(data[0].min() - 1, data[0].max() + 1, 100)
    y_grid = np.linspace(data[1].min() - 1, data[1].max() + 1, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(positions).reshape(X.shape)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data[0], data[1], s=5, alpha=0.3, c="black")
    contour = ax.contour(X, Y, Z, levels=10, cmap="viridis")
    ax.clabel(contour, inline=True, fontsize=8)
    plt.colorbar(contour, label="density")
    plt.xlabel("CSI:FingerIDScore")
    plt.ylabel("ionMass")
    plt.title("KDE contours over data")
    plt.show()
    return


@app.cell
def _(df):
    df["structurePerIdRank"].describe()
    return


@app.cell
def _(pl):
    pl.scan_parquet("structure_identifications_all.parquet").head().collect()
    return


@app.cell
def _(pl):
    pl.scan_parquet("structure_identifications_all.parquet").group_by("mappingFeatureId").agg(
        pl.col("CSI:FingerIDScore").mean().alias("mean"),
        pl.col("CSI:FingerIDScore").median().alias("median"),
        pl.col("CSI:FingerIDScore").var().alias("var"),
        pl.col("CSI:FingerIDScore").min().alias("min"),
        pl.col("CSI:FingerIDScore").max().alias("max"),
        pl.col("CSI:FingerIDScore").count().alias("count"),
    ).collect()
    return


if __name__ == "__main__":
    app.run()
