import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import matplotlib.pyplot as plt
    import polars as pl
    import seaborn as sns

    return pl, sns


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
        pl.col("fold")
    )

    df = (
        pl.read_parquet("structure_identifications_all.parquet")
        .sample(fraction=0.3, seed=42)
        .join(inchikey_map, on="mappingFeatureId", how="left")
        .with_columns(
            # In polars `==` propagates null, so unmatched keys would give null
            # rather than 0. pandas compares NaN as False, hence fill_null(False).
            (pl.col("InChIkey2D") == pl.col("inchikey_msg"))
            .fill_null(False)
            .cast(pl.Int8)
            .alias("is_correct")
        )
        # `isfinite | isnan` keeps everything except +/-inf. is_infinite() is
        # null for null rows, so fill_null(False) keeps missing scores as pandas did.
        .filter(~pl.col("CSI:FingerIDScore").is_infinite().fill_null(False))
    )
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(df, pl, sns):
    sns.histplot(
        data=df.group_by(["inchikey_msg", "InChIkey2D"])
        .agg(pl.col("CSI:FingerIDScore").mean(), pl.col("ConfidenceScoreExact").mean())
        .with_columns(
            (pl.col("InChIkey2D") == pl.col("inchikey_msg"))
            .fill_null(False)
            .cast(pl.Int8)
            .alias("is_correct")
        ),
        x="CSI:FingerIDScore",
        y="ConfidenceScoreExact",
        hue="is_correct",
        stat="density",
        common_norm=False,
        kde=True,
    )
    return


@app.cell
def _(df, pl, sns):
    sns.histplot(
        data=df.filter(pl.col("ionMass")>500.0),
        x="CSI:FingerIDScore",
        y="ionMass",
        hue="is_correct",
    )
    return


if __name__ == "__main__":
    app.run()
