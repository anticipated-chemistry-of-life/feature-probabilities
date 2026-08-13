import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import matplotlib.pyplot as plt
    import polars as pl
    import seaborn as sns
    import numpy as np

    return pl, sns


@app.cell
def _(pl):
    df = (
        pl.scan_parquet("cfmid_scores.parquet")
        .with_columns(
            (pl.col("inchikey_isdb") == pl.col("inchikey_msg"))
            .cast(pl.Int8)
            .alias("is_correct")
        )
        .filter(pl.col("entropy_similarity") > 0.0)
        .collect()
    )
    return (df,)


@app.cell
def _(df, sns):
    sns.histplot(
        data=df,
        x="entropy_similarity",
        hue="is_correct",
        common_norm=False,
        stat="density",
        kde=True,
    )
    return


@app.cell
def _(pl, sns):
    _df = (
        pl.scan_parquet("cfmid_scores.parquet")
        .with_columns(
            (pl.col("inchikey_isdb") == pl.col("inchikey_msg"))
            .cast(pl.Int8)
            .alias("is_correct")
        )
        .filter(pl.col("entropy_similarity") > 0.0)
        .group_by("identifier")
        .agg(
            pl.col("entropy_similarity").mean().alias("mean"),
            pl.col("entropy_similarity").median().alias("median"),
            pl.col("entropy_similarity").var().alias("var"),
            pl.col("entropy_similarity").min().alias("min"),
            pl.col("entropy_similarity").max().alias("max"),
            pl.col("entropy_similarity").count().alias("count"),
            (pl.col("is_correct").sum() > 0)
            .cast(pl.Int8)
            .alias("correct_mol_was_found"),
            pl.col("entropy_similarity")
            .sort(descending=True)
            .head(2)
            .diff()
            .drop_nulls()
            .first()
            .fill_null(0)
            .abs()
            .alias("top1_top2_diff"),
        )
        .collect()
    )

    sns.histplot(
        data=_df,
        x="max",
        hue="correct_mol_was_found",
        common_norm=False,
        stat="density",
        kde=True,
    )
    return


@app.cell
def _(pl, sns):
    _df = (
        pl.scan_parquet("cfmid_scores.parquet")
        .with_columns(
            (pl.col("inchikey_isdb") == pl.col("inchikey_msg"))
            .cast(pl.Int8)
            .alias("is_correct")
        )
        .collect()
    )

    sns.histplot(
        data=_df,
        hue="is_correct",
        x="entropy_similarity",
        y="ModifiedCosineGreedy",
        common_norm=False,
    )
    return


if __name__ == "__main__":
    app.run()
