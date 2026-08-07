from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import polars as pl

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

WINDOW_WEEKS = 16
VAL_DAYS = 7
N_CUSTOMERS = 20_000
MIN_ARTICLE_TRANSACTIONS = 5
MAX_ARTICLES = 15_000
TITLE_MAX_CHARS = 200


def _age_band_expr() -> pl.Expr:
    age = pl.col("age")
    return (
        pl.when(age.is_null())
        .then(pl.lit("unknown"))
        .when(age < 18)
        .then(pl.lit("under_18"))
        .when(age <= 25)
        .then(pl.lit("18_25"))
        .when(age <= 35)
        .then(pl.lit("26_35"))
        .when(age <= 50)
        .then(pl.lit("36_50"))
        .otherwise(pl.lit("over_50"))
        .alias("age_band")
    )


def build_samples(raw_dir: Path = RAW_DIR) -> dict[str, pl.DataFrame]:
    transactions = pl.scan_csv(raw_dir / "transactions_train.csv").with_columns(
        pl.col("t_dat").str.to_date("%Y-%m-%d")
    )

    max_date = transactions.select(pl.col("t_dat").max()).collect().item()
    window_start = max_date - timedelta(weeks=WINDOW_WEEKS)
    windowed = transactions.filter(pl.col("t_dat") >= window_start)

    top_customers = (
        windowed.group_by("customer_id")
        .agg(pl.len().alias("n_tx"))
        .sort("n_tx", descending=True)
        .head(N_CUSTOMERS)
        .collect()
    )

    by_customers = windowed.filter(pl.col("customer_id").is_in(top_customers["customer_id"].to_list()))

    top_articles = (
        by_customers.group_by("article_id")
        .agg(pl.len().alias("n_tx"))
        .filter(pl.col("n_tx") >= MIN_ARTICLE_TRANSACTIONS)
        .sort("n_tx", descending=True)
        .head(MAX_ARTICLES)
        .collect()
    )

    sampled = by_customers.filter(pl.col("article_id").is_in(top_articles["article_id"].to_list())).collect()

    val_start = max_date - timedelta(days=VAL_DAYS - 1)
    val = sampled.filter(pl.col("t_dat") >= val_start)
    train = sampled.filter(pl.col("t_dat") < val_start)

    val_only_customers = set(val["customer_id"]) - set(train["customer_id"])
    print(f"customers in val but absent from train: {len(val_only_customers)}")

    articles = pl.scan_csv(raw_dir / "articles.csv")
    items = (
        articles.filter(pl.col("article_id").is_in(top_articles["article_id"].to_list()))
        .with_columns(
            (pl.col("prod_name").fill_null("") + pl.lit(" ") + pl.col("detail_desc").fill_null(""))
            .str.slice(0, TITLE_MAX_CHARS)
            .alias("title"),
            pl.col("product_group_name").alias("category_l1"),
            pl.col("product_type_name").alias("category_l2"),
            pl.col("colour_group_name").alias("colour"),
            pl.col("department_name").alias("dept"),
        )
        .select("article_id", "title", "category_l1", "category_l2", "colour", "dept")
        .collect()
        .join(
            top_articles.select("article_id", pl.col("n_tx").alias("n_interactions")),
            on="article_id",
            how="left",
        )
    )

    customers = pl.scan_csv(raw_dir / "customers.csv")
    users = (
        customers.filter(pl.col("customer_id").is_in(top_customers["customer_id"].to_list()))
        .with_columns(
            _age_band_expr(),
            pl.col("postal_code").str.slice(0, 2).alias("region"),
            pl.col("club_member_status").alias("club_status"),
        )
        .select("customer_id", "age_band", "region", "club_status")
        .collect()
    )

    return {"train": train, "val": val, "items": items, "users": users}


def write_samples(samples: dict[str, pl.DataFrame], processed_dir: Path = PROCESSED_DIR) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in samples.items():
        frame.write_parquet(processed_dir / f"{name}.parquet")


def _report(samples: dict[str, pl.DataFrame]) -> None:
    for name in ("train", "val", "items", "users"):
        frame = samples[name]
        if "t_dat" in frame.columns and frame.height > 0:
            date_range = f"{frame['t_dat'].min()} to {frame['t_dat'].max()}"
        else:
            date_range = "n/a"
        print(f"{name}: {frame.height} rows, date range {date_range}")


def main() -> None:
    samples = build_samples()
    write_samples(samples)
    _report(samples)


if __name__ == "__main__":
    main()
