from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_GAMES_CSV,
    DEFAULT_RECOMMENDATIONS_CSV,
    GAMES_CLEAN_CSV,
    MODEL_PREPARED_CSV,
    PCA_COMPONENTS_CSV,
    PCA_EXPLAINED_VARIANCE_CSV,
    PIPELINE_REPORT_JSON,
    RECOMMENDATIONS_AGG_CSV,
    REVIEW_TRENDS_CSV,
    STEAM_ANALYTICS_CSV,
    STEAM_CATALOG_API_CSV,
    ensure_directories,
)
from src.utils import (
    clean_column_name,
    minmax,
    oversample_minority,
    parse_owner_range,
    safe_divide,
    split_multi_value,
    standardize,
    top_values,
)


GAME_COLUMNS = [
    "AppID",
    "Name",
    "Release date",
    "Estimated owners",
    "Peak CCU",
    "Required age",
    "Price",
    "Discount",
    "DLC count",
    "Windows",
    "Mac",
    "Linux",
    "Metacritic score",
    "User score",
    "Positive",
    "Negative",
    "Achievements",
    "Recommendations",
    "Average playtime forever",
    "Average playtime two weeks",
    "Median playtime forever",
    "Median playtime two weeks",
    "Developers",
    "Publishers",
    "Categories",
    "Genres",
    "Tags",
]

NUMERIC_GAME_COLUMNS = [
    "Peak CCU",
    "Required age",
    "Price",
    "Discount",
    "DLC count",
    "Metacritic score",
    "User score",
    "Positive",
    "Negative",
    "Achievements",
    "Recommendations",
    "Average playtime forever",
    "Average playtime two weeks",
    "Median playtime forever",
    "Median playtime two weeks",
]


def read_games_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))

    fixed_header: list[str] = []
    for column in header:
        if column == "DiscountDLC count":
            fixed_header.extend(["Discount", "DLC count"])
        else:
            fixed_header.append(column)
    return fixed_header


def load_and_clean_games(path: Path) -> tuple[pd.DataFrame, dict]:
    fixed_header = read_games_header(path)
    games = pd.read_csv(
        path,
        names=fixed_header,
        header=None,
        skiprows=1,
        usecols=GAME_COLUMNS,
        low_memory=False,
        encoding="utf-8-sig",
    )

    original_missing = games.isna().sum().to_dict()

    rename_map = {
        "AppID": "app_id",
        "Name": "name",
        "Release date": "release_date",
        "Estimated owners": "estimated_owners",
        "Peak CCU": "peak_ccu",
        "Required age": "required_age",
        "Price": "price",
        "Discount": "discount",
        "DLC count": "dlc_count",
        "Windows": "windows",
        "Mac": "mac",
        "Linux": "linux",
        "Metacritic score": "metacritic_score",
        "User score": "user_score",
        "Positive": "positive",
        "Negative": "negative",
        "Achievements": "achievements",
        "Recommendations": "steam_recommendations",
        "Average playtime forever": "avg_playtime_forever",
        "Average playtime two weeks": "avg_playtime_2weeks",
        "Median playtime forever": "median_playtime_forever",
        "Median playtime two weeks": "median_playtime_2weeks",
        "Developers": "developers",
        "Publishers": "publishers",
        "Categories": "categories",
        "Genres": "genres",
        "Tags": "tags",
    }
    games = games.rename(columns=rename_map)

    numeric_columns = [rename_map[col] for col in NUMERIC_GAME_COLUMNS]
    for column in numeric_columns:
        games[column] = pd.to_numeric(games[column], errors="coerce")

    for column in ["windows", "mac", "linux"]:
        games[column] = (
            games[column]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": 1, "false": 0})
            .fillna(0)
            .astype(int)
        )

    text_columns = ["developers", "publishers", "categories", "genres", "tags"]
    for column in text_columns:
        games[f"missing_{column}"] = games[column].isna().astype(int)
        games[column] = games[column].fillna("Unknown").replace("", "Unknown")

    games["name"] = games["name"].fillna("Unknown")
    games["release_date"] = pd.to_datetime(games["release_date"], errors="coerce")
    games["release_year"] = games["release_date"].dt.year
    games["release_month"] = games["release_date"].dt.month

    owners = parse_owner_range(games["estimated_owners"])
    games = pd.concat([games, owners], axis=1)
    games["owners_mid"] = games["owners_mid"].fillna(games["owners_mid"].median())

    games["price_imputed"] = games["price"].fillna(games["price"].median())
    games["required_age"] = games["required_age"].fillna(0)
    games["discount"] = games["discount"].fillna(0)
    games["dlc_count"] = games["dlc_count"].fillna(0)

    games["has_metacritic"] = (games["metacritic_score"].fillna(0) > 0).astype(int)
    metacritic_nonzero = games.loc[games["metacritic_score"] > 0, "metacritic_score"]
    metacritic_median = float(metacritic_nonzero.median()) if len(metacritic_nonzero) else 0.0
    games["metacritic_score_imputed"] = games["metacritic_score"].where(
        games["metacritic_score"] > 0, metacritic_median
    )

    games["positive"] = games["positive"].fillna(0)
    games["negative"] = games["negative"].fillna(0)
    games["total_steam_reviews"] = games["positive"] + games["negative"]
    games["steam_positive_rate"] = safe_divide(
        games["positive"], games["total_steam_reviews"]
    )
    games["steam_positive_rate"] = games["steam_positive_rate"].fillna(0)

    games["is_free"] = (games["price_imputed"] == 0).astype(int)
    games["platform_count"] = games[["windows", "mac", "linux"]].sum(axis=1)
    games["primary_genre"] = games["genres"].apply(
        lambda value: split_multi_value(value)[0] if split_multi_value(value) else "Unknown"
    )

    games["price_band"] = pd.cut(
        games["price_imputed"],
        bins=[-0.01, 0, 5, 15, 30, np.inf],
        labels=["Free", "Low", "Medium", "High", "Premium"],
        include_lowest=True,
    ).astype(str)

    games["review_volume_band"] = pd.qcut(
        games["total_steam_reviews"].rank(method="first"),
        q=5,
        labels=["Very low", "Low", "Medium", "High", "Very high"],
    ).astype(str)

    games = games.drop_duplicates(subset=["app_id"]).reset_index(drop=True)

    report = {
        "rows": int(len(games)),
        "unique_app_ids": int(games["app_id"].nunique()),
        "header_fix": "Original column 'DiscountDLC count' was split into 'Discount' and 'DLC count'.",
        "missing_before_imputation": {str(k): int(v) for k, v in original_missing.items()},
        "imputation": {
            "text_columns": "Missing text values were filled with 'Unknown' and missing flags were kept.",
            "price": "Missing prices were filled with the median.",
            "metacritic": "Zero or missing Metacritic values were replaced by the nonzero median in a separate imputed column.",
        },
    }
    return games, report


def aggregate_recommendations(
    path: Path,
    chunksize: int = 1_000_000,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    app_parts: list[pd.DataFrame] = []
    trend_parts: list[pd.DataFrame] = []
    rows_read = 0

    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        if max_rows is not None:
            remaining = max_rows - rows_read
            if remaining <= 0:
                break
            chunk = chunk.iloc[:remaining].copy()

        rows_read += len(chunk)
        chunk["app_id"] = pd.to_numeric(chunk["app_id"], errors="coerce")
        chunk = chunk.dropna(subset=["app_id"])
        chunk["app_id"] = chunk["app_id"].astype(int)
        chunk["helpful"] = pd.to_numeric(chunk["helpful"], errors="coerce").fillna(0)
        chunk["funny"] = pd.to_numeric(chunk["funny"], errors="coerce").fillna(0)
        chunk["hours"] = pd.to_numeric(chunk["hours"], errors="coerce").fillna(0)
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        chunk["year"] = chunk["date"].dt.year
        chunk["recommended_flag"] = (
            chunk["is_recommended"].astype(str).str.lower().eq("true").astype(int)
        )

        app_group = chunk.groupby("app_id", as_index=False).agg(
            recommendation_count=("review_id", "count"),
            recommended_count=("recommended_flag", "sum"),
            hours_sum=("hours", "sum"),
            hours_count=("hours", "count"),
            helpful_sum=("helpful", "sum"),
            funny_sum=("funny", "sum"),
            unique_review_users=("user_id", "nunique"),
            first_review_date=("date", "min"),
            last_review_date=("date", "max"),
        )
        app_parts.append(app_group)

        trend_group = chunk.dropna(subset=["year"]).groupby("year", as_index=False).agg(
            review_count=("review_id", "count"),
            recommended_count=("recommended_flag", "sum"),
            hours_sum=("hours", "sum"),
        )
        trend_parts.append(trend_group)

        if max_rows is not None and rows_read >= max_rows:
            break

    if not app_parts:
        raise ValueError("No recommendation rows were loaded.")

    app_all = pd.concat(app_parts, ignore_index=True)
    app_agg = app_all.groupby("app_id", as_index=False).agg(
        recommendation_count=("recommendation_count", "sum"),
        recommended_count=("recommended_count", "sum"),
        hours_sum=("hours_sum", "sum"),
        hours_count=("hours_count", "sum"),
        helpful_sum=("helpful_sum", "sum"),
        funny_sum=("funny_sum", "sum"),
        unique_review_users=("unique_review_users", "sum"),
        first_review_date=("first_review_date", "min"),
        last_review_date=("last_review_date", "max"),
    )
    app_agg["not_recommended_count"] = (
        app_agg["recommendation_count"] - app_agg["recommended_count"]
    )
    app_agg["external_recommendation_rate"] = safe_divide(
        app_agg["recommended_count"], app_agg["recommendation_count"]
    )
    app_agg["avg_review_hours"] = safe_divide(app_agg["hours_sum"], app_agg["hours_count"])

    trends = pd.concat(trend_parts, ignore_index=True)
    trends = trends.groupby("year", as_index=False).agg(
        review_count=("review_count", "sum"),
        recommended_count=("recommended_count", "sum"),
        hours_sum=("hours_sum", "sum"),
    )
    trends["recommendation_rate"] = safe_divide(
        trends["recommended_count"], trends["review_count"]
    )
    trends["year"] = trends["year"].astype(int)

    report = {
        "rows_read": int(rows_read),
        "unique_app_ids": int(app_agg["app_id"].nunique()),
        "date_min": str(app_agg["first_review_date"].min().date()),
        "date_max": str(app_agg["last_review_date"].max().date()),
        "recommended_rate_global": float(
            app_agg["recommended_count"].sum() / app_agg["recommendation_count"].sum()
        ),
    }
    return app_agg, trends, report


def add_catalog_bonus_if_available(analytics: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if not STEAM_CATALOG_API_CSV.exists():
        analytics["catalog_api_match"] = 0
        return analytics, {
            "used": False,
            "reason": "Run python -m src.collect_steam_catalog before preparing data to generate the bonus API file.",
        }

    catalog = pd.read_csv(STEAM_CATALOG_API_CSV, low_memory=False)
    if "appid" not in catalog.columns:
        return analytics, {"used": False, "reason": "API catalog file did not contain appid."}

    catalog = catalog.rename(
        columns={
            "appid": "app_id",
            "name": "catalog_api_name",
            "owners": "catalog_api_owners",
            "ccu": "catalog_api_ccu",
            "score_rank": "catalog_api_score_rank",
        }
    )
    catalog = catalog.drop_duplicates(subset=["app_id"])
    keep = [
        col
        for col in [
            "app_id",
            "catalog_api_name",
            "catalog_api_owners",
            "catalog_api_ccu",
            "catalog_api_score_rank",
            "data_coleta_utc",
            "source_api",
        ]
        if col in catalog
    ]
    analytics = analytics.merge(catalog[keep], on="app_id", how="left")
    analytics["catalog_api_match"] = analytics["catalog_api_name"].notna().astype(int)
    return analytics, {
        "used": True,
        "catalog_rows": int(len(catalog)),
        "matched_games": int(analytics["catalog_api_match"].sum()),
    }


def add_transformations(analytics: pd.DataFrame) -> pd.DataFrame:
    analytics["recommendation_count"] = analytics["recommendation_count"].fillna(0)
    analytics["recommended_count"] = analytics["recommended_count"].fillna(0)
    analytics["not_recommended_count"] = analytics["not_recommended_count"].fillna(0)
    analytics["avg_review_hours"] = analytics["avg_review_hours"].fillna(0)
    analytics["external_recommendation_rate"] = analytics[
        "external_recommendation_rate"
    ].fillna(0)
    analytics["helpful_sum"] = analytics["helpful_sum"].fillna(0)
    analytics["funny_sum"] = analytics["funny_sum"].fillna(0)

    analytics["hours_band"] = pd.cut(
        analytics["avg_review_hours"],
        bins=[-0.01, 1, 10, 50, 150, np.inf],
        labels=["0-1h", "1-10h", "10-50h", "50-150h", "150h+"],
        include_lowest=True,
    ).astype(str)
    analytics["recommendation_rate_band"] = pd.cut(
        analytics["external_recommendation_rate"],
        bins=[-0.01, 0.5, 0.75, 0.9, 1.0],
        labels=["Low", "Medium", "High", "Excellent"],
        include_lowest=True,
    ).astype(str)

    numeric_features = [
        "price_imputed",
        "owners_mid",
        "peak_ccu",
        "total_steam_reviews",
        "recommendation_count",
        "avg_review_hours",
        "helpful_sum",
    ]
    for column in numeric_features:
        analytics[column] = pd.to_numeric(analytics[column], errors="coerce").fillna(0)
        log_column = f"log_{column}"
        analytics[log_column] = np.log1p(analytics[column].clip(lower=0))
        analytics[f"norm_{log_column}"] = minmax(analytics[log_column])
        analytics[f"std_{log_column}"] = standardize(analytics[log_column])

    analytics["popularity_score"] = (
        analytics["norm_log_owners_mid"] * 0.30
        + analytics["norm_log_total_steam_reviews"] * 0.25
        + analytics["norm_log_recommendation_count"] * 0.25
        + analytics["norm_log_peak_ccu"] * 0.20
    )
    return analytics


def build_model_dataset(
    analytics: pd.DataFrame, top_n_genres: int = 12
) -> tuple[pd.DataFrame, dict, pd.DataFrame, list[str]]:
    top_genres = top_values(analytics["genres"], top_n=top_n_genres)
    model = analytics[
        [
            "app_id",
            "name",
            "price_band",
            "primary_genre",
            "windows",
            "mac",
            "linux",
            "platform_count",
            "is_free",
            "external_recommendation_rate",
            "recommendation_count",
            "popularity_score",
            "norm_log_price_imputed",
            "std_log_price_imputed",
            "norm_log_owners_mid",
            "std_log_owners_mid",
            "norm_log_total_steam_reviews",
            "std_log_total_steam_reviews",
            "norm_log_avg_review_hours",
            "std_log_avg_review_hours",
        ]
    ].copy()

    for genre in top_genres:
        col = f"genre_{clean_column_name(genre)}"
        model[col] = analytics["genres"].apply(lambda value: int(genre in split_multi_value(value)))

    price_dummies = pd.get_dummies(model["price_band"], prefix="price_band", dtype=int)
    model = pd.concat([model, price_dummies], axis=1)

    model["target_high_recommendation"] = (
        (model["external_recommendation_rate"] >= 0.85)
        & (model["recommendation_count"] >= 50)
    ).astype(int)

    candidate_features = [
        col
        for col in model.columns
        if col.startswith(("norm_", "std_", "genre_", "price_band_"))
        or col in ["windows", "mac", "linux", "platform_count", "is_free", "popularity_score"]
    ]
    variances = model[candidate_features].var(numeric_only=True)
    kept_features = variances[variances > 0.0001].index.tolist()
    dropped_low_variance = variances[variances <= 0.0001].index.tolist()

    final_columns = ["app_id", "name", "target_high_recommendation"] + kept_features
    model_selected = model[final_columns].copy()
    balanced, oversampling_report = oversample_minority(
        model_selected, "target_high_recommendation"
    )

    report = {
        "top_genres_encoded": top_genres,
        "candidate_features": candidate_features,
        "kept_features": kept_features,
        "dropped_low_variance_features": dropped_low_variance,
        "oversampling": oversampling_report,
    }
    return balanced, report, model_selected, kept_features


def build_pca_outputs(
    model_selected: pd.DataFrame,
    analytics: pd.DataFrame,
    feature_columns: list[str],
    n_components: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Apply PCA with NumPy SVD over encoded and scaled model features."""
    if not feature_columns:
        raise ValueError("PCA requires at least one numeric feature.")

    x = model_selected[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    x_values = x.to_numpy(dtype=float)
    means = x_values.mean(axis=0)
    stds = x_values.std(axis=0)
    stds[stds == 0] = 1
    x_scaled = (x_values - means) / stds

    _, singular_values, vt = np.linalg.svd(x_scaled, full_matrices=False)
    component_count = min(n_components, vt.shape[0])
    loadings = vt[:component_count].T
    scores = x_scaled @ loadings

    if len(x_scaled) > 1:
        explained_variance = (singular_values**2) / (len(x_scaled) - 1)
    else:
        explained_variance = singular_values**2
    total_variance = explained_variance.sum()
    explained_ratio = (
        explained_variance / total_variance if total_variance > 0 else np.zeros_like(explained_variance)
    )

    labels = analytics[
        [
            "app_id",
            "name",
            "primary_genre",
            "price_band",
            "recommendation_count",
            "external_recommendation_rate",
            "avg_review_hours",
            "popularity_score",
        ]
    ].drop_duplicates(subset=["app_id"])
    components = model_selected[["app_id", "target_high_recommendation"]].merge(
        labels, on="app_id", how="left"
    )
    for idx in range(component_count):
        components[f"PC{idx + 1}"] = scores[:, idx]

    variance_rows = []
    cumulative = 0.0
    for idx in range(component_count):
        cumulative += float(explained_ratio[idx])
        variance_rows.append(
            {
                "component": f"PC{idx + 1}",
                "explained_variance": float(explained_variance[idx]),
                "explained_variance_ratio": float(explained_ratio[idx]),
                "cumulative_explained_variance_ratio": cumulative,
            }
        )
    variance = pd.DataFrame(variance_rows)

    top_loading_rows = []
    for idx in range(component_count):
        loading_series = pd.Series(loadings[:, idx], index=feature_columns)
        top_features = loading_series.abs().sort_values(ascending=False).head(6).index
        top_loading_rows.append(
            {
                "component": f"PC{idx + 1}",
                "top_features": [
                    {"feature": feature, "loading": float(loading_series[feature])}
                    for feature in top_features
                ],
            }
        )

    report = {
        "method": "PCA implemented with NumPy SVD over centered and standardized encoded features.",
        "n_components": int(component_count),
        "features_used": feature_columns,
        "explained_variance_ratio": {
            f"PC{idx + 1}": float(explained_ratio[idx]) for idx in range(component_count)
        },
        "cumulative_explained_variance_ratio": float(variance["cumulative_explained_variance_ratio"].iloc[-1]),
        "top_loadings": top_loading_rows,
    }
    return components, variance, report


def run_pipeline(
    games_csv: Path = DEFAULT_GAMES_CSV,
    recommendations_csv: Path = DEFAULT_RECOMMENDATIONS_CSV,
    chunksize: int = 1_000_000,
    max_recommendation_rows: int | None = None,
) -> dict:
    ensure_directories()
    games, games_report = load_and_clean_games(games_csv)
    games.to_csv(GAMES_CLEAN_CSV, index=False)

    recs, trends, recs_report = aggregate_recommendations(
        recommendations_csv,
        chunksize=chunksize,
        max_rows=max_recommendation_rows,
    )
    recs.to_csv(RECOMMENDATIONS_AGG_CSV, index=False)
    trends.to_csv(REVIEW_TRENDS_CSV, index=False)

    analytics = games.merge(recs, on="app_id", how="left")
    analytics, catalog_report = add_catalog_bonus_if_available(analytics)
    analytics = add_transformations(analytics)
    analytics.to_csv(STEAM_ANALYTICS_CSV, index=False)

    model, model_report, model_selected, pca_features = build_model_dataset(analytics)
    model.to_csv(MODEL_PREPARED_CSV, index=False)

    pca_components, pca_variance, pca_report = build_pca_outputs(
        model_selected, analytics, pca_features
    )
    pca_components.to_csv(PCA_COMPONENTS_CSV, index=False)
    pca_variance.to_csv(PCA_EXPLAINED_VARIANCE_CSV, index=False)

    overlap = int(analytics["recommendation_count"].fillna(0).gt(0).sum())
    report = {
        "source_files": {
            "games_csv": str(games_csv),
            "recommendations_csv": str(recommendations_csv),
            "steam_catalog_api_csv": str(STEAM_CATALOG_API_CSV),
        },
        "games": games_report,
        "recommendations": recs_report,
        "integration": {
            "games_with_recommendations": overlap,
            "games_total": int(len(analytics)),
            "games_with_recommendations_pct": round(100 * overlap / len(analytics), 2),
        },
        "transformations": {
            "normalization": "Min-max columns start with norm_log_*. Skewed numeric values were transformed with log1p first.",
            "standardization": "Z-score columns start with std_log_*.",
            "encoding": "Top genres and price bands were encoded in the model_prepared_oversampled.csv file.",
            "discretization": "price_band, review_volume_band, hours_band and recommendation_rate_band were created.",
            "oversampling": "A supervised preparation table balances target_high_recommendation using random oversampling.",
            "feature_selection": "Very low variance encoded features were removed before oversampling.",
            "pca": "PCA was applied with NumPy SVD over the encoded/scaled model features, generating PC1, PC2 and PC3.",
        },
        "catalog_bonus": catalog_report,
        "model_preparation": model_report,
        "pca": pca_report,
        "outputs": {
            "games_clean": str(GAMES_CLEAN_CSV),
            "recommendations_aggregated": str(RECOMMENDATIONS_AGG_CSV),
            "review_trends": str(REVIEW_TRENDS_CSV),
            "steam_analytics": str(STEAM_ANALYTICS_CSV),
            "model_prepared_oversampled": str(MODEL_PREPARED_CSV),
            "pca_components": str(PCA_COMPONENTS_CSV),
            "pca_explained_variance": str(PCA_EXPLAINED_VARIANCE_CSV),
        },
    }
    PIPELINE_REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Steam dashboard datasets.")
    parser.add_argument("--games-csv", type=Path, default=DEFAULT_GAMES_CSV)
    parser.add_argument("--recommendations-csv", type=Path, default=DEFAULT_RECOMMENDATIONS_CSV)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument(
        "--max-recommendation-rows",
        type=int,
        default=None,
        help="Optional limit for quick test runs. Omit for full processing.",
    )
    args = parser.parse_args()
    report = run_pipeline(
        games_csv=args.games_csv,
        recommendations_csv=args.recommendations_csv,
        chunksize=args.chunksize,
        max_recommendation_rows=args.max_recommendation_rows,
    )
    print(json.dumps(report["integration"], indent=2))


if __name__ == "__main__":
    main()
