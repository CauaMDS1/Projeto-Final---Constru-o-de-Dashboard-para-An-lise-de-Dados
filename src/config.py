from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

LOCAL_GAMES_CSV = RAW_DIR / "games.csv"
LOCAL_RECOMMENDATIONS_CSV = RAW_DIR / "recommendations.csv"

DEFAULT_GAMES_CSV = Path(os.environ.get("GAMES_CSV", str(LOCAL_GAMES_CSV)))
DEFAULT_RECOMMENDATIONS_CSV = Path(
    os.environ.get("RECOMMENDATIONS_CSV", str(LOCAL_RECOMMENDATIONS_CSV))
)

STEAM_CATALOG_API_CSV = RAW_DIR / "steam_catalog_api.csv"
GAMES_CLEAN_CSV = PROCESSED_DIR / "games_clean.csv"
RECOMMENDATIONS_AGG_CSV = PROCESSED_DIR / "recommendations_aggregated.csv"
REVIEW_TRENDS_CSV = PROCESSED_DIR / "review_trends_by_year.csv"
STEAM_ANALYTICS_CSV = PROCESSED_DIR / "steam_analytics.csv"
MODEL_PREPARED_CSV = PROCESSED_DIR / "model_prepared_oversampled.csv"
PCA_COMPONENTS_CSV = PROCESSED_DIR / "pca_components.csv"
PCA_EXPLAINED_VARIANCE_CSV = PROCESSED_DIR / "pca_explained_variance.csv"
PIPELINE_REPORT_JSON = PROCESSED_DIR / "pipeline_report.json"


def ensure_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
