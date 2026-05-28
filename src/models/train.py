"""
src/models/train.py
=============================================================================
Model Factory — LightGBM Grid Imbalance Regressor
=============================================================================
Responsibilities:
  - Load the Gold layer feature matrix from Parquet
  - Perform a chronologically-safe walk-forward cross-validation using
    scikit-learn's TimeSeriesSplit to select hyperparameters and report
    out-of-fold generalisation metrics
  - Execute a final hold-out train/test split (last N hours = test set)
    that is never touched during cross-validation
  - Fit a production LightGBM Regressor on the full training window
  - Serialise the trained model artifact (+ feature list + training metadata)
    to outputs/models/ using joblib for later live-inference loading

Target variable: imbalance_mwh  (Net Grid Imbalance Volume, MWh)

TimeSeriesSplit rationale
──────────────────────────
Standard k-fold randomly shuffles observations.  For time-series data this
causes future information to leak into training folds — a model that "knows"
next week's imbalance from a shuffled training row will appear superhuman on
CV but fail catastrophically on fresh data.

TimeSeriesSplit enforces expanding-window folds:
  Fold 1:  train=[t0..t1]        test=[t1..t2]
  Fold 2:  train=[t0..t2]        test=[t2..t3]
  Fold N:  train=[t0..t(N-1)]    test=[t(N-1)..tN]

Each test fold is strictly in the future relative to its training fold,
mirroring real-world forecasting conditions.

Architecture position: GOLD LAYER → MODEL FACTORY → outputs/models/
Upstream:  src/features/grid_balancing.py  (GridBalancingFeatureEngineer)
Downstream: src/models/evaluate.py, live inference
=============================================================================
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ..utils.pipeline_logging import get_pipeline_logger

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logger = get_pipeline_logger("model_factory.train")


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class TrainingConfig:
    """
    Immutable configuration for one training run.

    Serialised alongside the model artifact so any future inference session
    can reconstruct the exact feature set and hyperparameters used.

    Attributes
    ──────────
    gold_parquet_dir:
        Root directory of the Gold layer Parquet store.
        Expects zone-partitioned sub-directories: zone=SE1/, zone=SE2/, ...
    zone_filter:
        Subset of bidding zones to train on.  None = all available zones.
        Each zone produces an independent model artifact.
    target_col:
        Target variable column name (must exist in Gold layer).
    test_hours:
        Number of hours reserved as the final hold-out test set.
        These rows are NEVER seen during cross-validation or training.
        Default: 720 h (≈ 30 days).
    n_cv_splits:
        Number of TimeSeriesSplit folds for cross-validation.
        Minimum recommended: 3.  Default: 5.
    cv_gap_hours:
        Gap inserted between each CV train/test fold to prevent adjacent-
        hour leakage.  Set to the maximum lag used in feature engineering
        (168 h = 7 days) so the test fold cannot "see" training features
        that overlap its window.
    lgbm_params:
        LightGBM hyperparameters passed to lgb.LGBMRegressor().
    early_stopping_rounds:
        LightGBM early stopping patience (requires a validation set).
        Applied on the final fit against a small intra-training validation
        window (last 10% of training rows).
    random_state:
        Seed for reproducibility.
    output_dir:
        Directory where model artifacts and metadata are written.
    """
    gold_parquet_dir: str = "data/gold"
    zone_filter: Optional[list[str]] = None
    target_col: str = "imbalance_mwh"
    #test_hours: int = 720
    test_hours: int = 168
    n_cv_splits: int = 5
    cv_gap_hours: int = 168          # 7-day gap between CV folds
    early_stopping_rounds: int = 50
    random_state: int = 42
    output_dir: str = "outputs/models"
    lgbm_params: dict = field(default_factory=lambda: {
        # Objective / loss
        "objective": "regression_l1",   # MAE loss — robust to imbalance spikes
        "metric": ["mae", "rmse"],
        # Tree structure
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "num_leaves": 127,
        "max_depth": -1,               # unlimited — num_leaves controls complexity
        "min_child_samples": 50,       # prevents overfitting on sparse winter weeks
        # Feature / row sampling (reduces overfitting, speeds training)
        "feature_fraction": 0.8,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        # Regularisation
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "min_gain_to_split": 0.01,
        # NaN handling (LightGBM supports NaN natively)
        "use_missing": True,
        # Verbosity
        "verbose": -1,
        "n_jobs": -1,
    })


# ===========================================================================
# Data Loading
# ===========================================================================

def load_gold_parquet(
    gold_dir: Path,
    zone_filter: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Load the Gold layer feature matrix from partitioned Parquet files.

    Expects the directory structure written by SilverLayerWriter /
    GridBalancingFeatureEngineer.write_gold():

        gold_dir/
          zone=SE1/feature_matrix.parquet
          zone=SE2/feature_matrix.parquet
          zone=SE3/feature_matrix.parquet
          zone=SE4/feature_matrix.parquet

    Falls back to a flat scan of all *.parquet files under gold_dir if the
    zone-partitioned layout is not found (e.g. all-zones single file).

    Args:
        gold_dir:    Root Gold layer directory.
        zone_filter: Optional list of zone names to load.  None = load all.

    Returns:
        Concatenated pd.DataFrame sorted by (zone, timestamp_utc).

    Raises:
        FileNotFoundError: If no Parquet files are found under gold_dir.
        ValueError:        If required columns are missing from the loaded data.
    """
    gold_dir = Path(gold_dir)
    if not gold_dir.exists():
        raise FileNotFoundError(
            f"Gold layer directory not found: {gold_dir}. "
            "Run GridBalancingFeatureEngineer.write_gold() first."
        )

    # Collect Parquet paths, optionally filtering by zone directory name
    parquet_files: list[Path] = []
    for p in sorted(gold_dir.rglob("*.parquet")):
        # If zone_filter is set, check that the file lives under zone=XX/
        if zone_filter:
            zone_dirs = {f"zone={z}" for z in zone_filter}
            if not any(part in zone_dirs for part in p.parts):
                continue
        parquet_files.append(p)

    if not parquet_files:
        raise FileNotFoundError(
            f"No Parquet files found under {gold_dir} "
            f"(zone_filter={zone_filter})."
        )

    logger.info("Loading %d Gold Parquet file(s) from %s", len(parquet_files), gold_dir)
    frames = [pd.read_parquet(p) for p in parquet_files]
    df = pd.concat(frames, ignore_index=True)

    # Ensure timestamp is UTC-aware
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df.sort_values(["zone", "timestamp_utc"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info(
        "Gold layer loaded | %d rows | %d columns | zones: %s | "
        "%s → %s",
        len(df), len(df.columns),
        sorted(df["zone"].unique().tolist()),
        df["timestamp_utc"].min(), df["timestamp_utc"].max(),
    )

    # Validate minimum required columns
    required = {"timestamp_utc", "zone", "imbalance_mwh", "feature_ready"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Gold layer DataFrame missing required columns: {missing}. "
            "Re-run GridBalancingFeatureEngineer.build_feature_matrix()."
        )

    return df


def resolve_feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    """
    Derive the model feature column list from Gold layer conventions.

    Excludes all identifier, target, metadata, and raw source columns.
    This mirrors the logic in GridBalancingFeatureEngineer.get_feature_columns()
    without requiring an engineer instance at inference time.

    Args:
        df:         Gold layer DataFrame.
        target_col: Target variable column name.

    Returns:
        Sorted list of feature column names.
    """
    _exclude = {
        # Identifiers
        "timestamp_utc", "zone", "year",
        # Targets
        target_col, "imbalance_mwh",
        # Raw source metrics (features use lag/rolling versions)
        "load_mw", "price_eur_mwh",
        "load_mw_clean", "imbalance_mwh_clean",
        # SCB / Riksbank raw levels
        "smahus_construction_index", "smahus_price_index",
        "riksbank_policy_rate_pct",
        # Quality flags and metadata
        "is_anomaly", "is_quarantined", "has_imputed_grid", "feature_ready",
        "direction", "resolution_minutes",
    }
    feature_cols = sorted([c for c in df.columns if c not in _exclude])
    logger.info(
        "Feature set resolved: %d columns (excluded %d non-feature cols)",
        len(feature_cols), len(_exclude),
    )
    return feature_cols


# ===========================================================================
# Train / Test Split
# ===========================================================================

def temporal_train_test_split(
    df: pd.DataFrame,
    test_hours: int,
    target_col: str,
    feature_cols: list[str],
    zone: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series,
           pd.DataFrame, pd.DataFrame]:
    """
    Perform a strictly chronological train/test split for one bidding zone.

    The test set is the LAST `test_hours` rows of the zone's feature-ready
    data.  It is set aside before any cross-validation or fitting and is
    only touched by evaluate.py to report final hold-out metrics.

    Args:
        df:           Full feature-ready Gold DataFrame for this zone.
        test_hours:   Number of hours to reserve as the hold-out test set.
        target_col:   Target column name.
        feature_cols: List of feature column names.
        zone:         Zone label for logging.

    Returns:
        X_train, y_train, X_test, y_test, meta_train, meta_test
        where meta_* contains timestamp_utc and zone for diagnostic plots.

    Raises:
        ValueError: If the dataset is too small to split.
    """
    # Filter to feature-ready rows only
    df_ready = df[df["feature_ready"]].copy()

    if len(df_ready) < test_hours * 2:
        raise ValueError(
            f"Zone {zone}: only {len(df_ready)} feature-ready rows available "
            f"({test_hours * 2} minimum required for a {test_hours}h test split). "
            "Extend the historical data window or reduce test_hours."
        )

    # Sort chronologically — critical; never shuffle
    df_ready = df_ready.sort_values("timestamp_utc").reset_index(drop=True)

    split_idx = len(df_ready) - test_hours
    df_train = df_ready.iloc[:split_idx]
    df_test  = df_ready.iloc[split_idx:]

    X_train = df_train[feature_cols]
    y_train = df_train[target_col]
    X_test  = df_test[feature_cols]
    y_test  = df_test[target_col]

    meta_train = df_train[["timestamp_utc", "zone"]].copy()
    meta_test  = df_test[["timestamp_utc", "zone"]].copy()

    logger.info(
        "Train/test split | zone=%s | train: %d rows (%s → %s) | "
        "test: %d rows (%s → %s)",
        zone,
        len(df_train),
        df_train["timestamp_utc"].min().date(),
        df_train["timestamp_utc"].max().date(),
        len(df_test),
        df_test["timestamp_utc"].min().date(),
        df_test["timestamp_utc"].max().date(),
    )
    return X_train, y_train, X_test, y_test, meta_train, meta_test


# ===========================================================================
# Cross-Validation
# ===========================================================================

def run_cross_validation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: TrainingConfig,
    zone: str,
) -> dict:
    """
    Walk-forward cross-validation using TimeSeriesSplit.

    Why TimeSeriesSplit and not standard KFold:
        Standard KFold randomly assigns rows to folds.  For a time series
        this means that fold N's training set may contain rows from hour
        T+500 while its test set contains rows from hour T+1 — the model
        is trained on the future and tested on the past, producing wildly
        optimistic CV scores that collapse on real data.

        TimeSeriesSplit enforces that train indices always precede test
        indices within every fold, mirroring the real-world forecasting
        setup where we predict forward from a fixed knowledge cutoff.

    Gap enforcement:
        cv_gap_hours rows are removed from the END of each training fold
        before fitting.  This prevents the model from using lag-168h
        features at hour T_test that were computed using data from hours
        [T_test - 168, T_test) — those hours would legitimately be in the
        training set but their "lags" could smuggle test-window information
        in if the gap is not enforced.

    Args:
        X_train: Training feature matrix (chronologically ordered).
        y_train: Training target series.
        config:  TrainingConfig instance.
        zone:    Zone label for logging.

    Returns:
        Dictionary of per-fold and aggregate CV metrics:
          {
            "fold_mae":  [float, ...],
            "fold_rmse": [float, ...],
            "mean_mae":  float,
            "mean_rmse": float,
            "std_mae":   float,
            "std_rmse":  float,
          }
    """
    tscv = TimeSeriesSplit(
        n_splits=config.n_cv_splits,
        gap=config.cv_gap_hours,         # enforced gap between train/test
    )

    fold_mae:  list[float] = []
    fold_rmse: list[float] = []

    logger.info(
        "Starting %d-fold TimeSeriesSplit CV | zone=%s | gap=%dh",
        config.n_cv_splits, zone, config.cv_gap_hours,
    )

    X_arr = X_train.values  # work on numpy arrays for speed
    y_arr = y_train.values

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_arr), start=1):
        X_fold_train, X_fold_val = X_arr[train_idx], X_arr[val_idx]
        y_fold_train, y_fold_val = y_arr[train_idx], y_arr[val_idx]

        # Small intra-fold validation set for early stopping (last 10% of fold train)
        val_split = max(1, int(len(X_fold_train) * 0.10))
        X_es_train = X_fold_train[:-val_split]
        y_es_train = y_fold_train[:-val_split]
        X_es_val   = X_fold_train[-val_split:]
        y_es_val   = y_fold_train[-val_split:]

        model = lgb.LGBMRegressor(
            **config.lgbm_params,
            random_state=config.random_state,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(
                X_es_train, y_es_train,
                eval_set=[(X_es_val, y_es_val)],
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=config.early_stopping_rounds,
                        verbose=False,
                    ),
                    lgb.log_evaluation(period=-1),  # suppress per-iter logs
                ],
            )

        y_pred = model.predict(X_fold_val)

        mae  = float(mean_absolute_error(y_fold_val, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_fold_val, y_pred)))
        fold_mae.append(mae)
        fold_rmse.append(rmse)

        logger.info(
            "CV Fold %d/%d | zone=%s | train=%d | val=%d | MAE=%.2f | RMSE=%.2f",
            fold_idx, config.n_cv_splits, zone,
            len(train_idx), len(val_idx), mae, rmse,
        )

    results = {
        "fold_mae":  fold_mae,
        "fold_rmse": fold_rmse,
        "mean_mae":  float(np.mean(fold_mae)),
        "mean_rmse": float(np.mean(fold_rmse)),
        "std_mae":   float(np.std(fold_mae)),
        "std_rmse":  float(np.std(fold_rmse)),
    }

    logger.info(
        "CV complete | zone=%s | MAE=%.2f ± %.2f MWh | RMSE=%.2f ± %.2f MWh",
        zone,
        results["mean_mae"], results["std_mae"],
        results["mean_rmse"], results["std_rmse"],
    )
    return results


# ===========================================================================
# Final Model Fit
# ===========================================================================

def fit_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: TrainingConfig,
    zone: str,
) -> lgb.LGBMRegressor:
    """
    Fit the production model on the full training set.

    Uses an intra-training validation split (last 10% of training rows)
    for early stopping so the model does not overfit.  This validation
    split is within the training window and does NOT touch the held-out
    test set.

    Args:
        X_train: Full training feature matrix.
        y_train: Full training target.
        config:  TrainingConfig instance.
        zone:    Zone label for logging.

    Returns:
        Fitted lgb.LGBMRegressor instance.
    """
    val_split = max(1, int(len(X_train) * 0.10))
    X_fit  = X_train.iloc[:-val_split]
    y_fit  = y_train.iloc[:-val_split]
    X_val  = X_train.iloc[-val_split:]
    y_val  = y_train.iloc[-val_split:]

    logger.info(
        "Fitting final model | zone=%s | fit=%d rows | "
        "early_stop_val=%d rows | n_estimators=%d",
        zone, len(X_fit), len(X_val), config.lgbm_params["n_estimators"],
    )

    model = lgb.LGBMRegressor(
        **config.lgbm_params,
        random_state=config.random_state,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            X_fit, y_fit,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=config.early_stopping_rounds,
                    verbose=False,
                ),
                lgb.log_evaluation(period=100),
            ],
        )

    best_iter = model.best_iteration_
    logger.info(
        "Final model fitted | zone=%s | best_iteration=%d",
        zone, best_iter if best_iter else config.lgbm_params["n_estimators"],
    )
    return model


# ===========================================================================
# Model Serialisation
# ===========================================================================

@dataclass
class ModelArtifact:
    """
    Container for everything needed to reload a model for live inference.

    Attributes
    ──────────
    model:          Fitted lgb.LGBMRegressor.
    feature_cols:   Ordered list of feature column names the model expects.
    zone:           Bidding zone this model was trained on.
    config:         TrainingConfig used for this training run.
    cv_metrics:     Cross-validation results dict.
    train_metadata: Provenance dict (timestamps, row counts, data window).
    """
    model: lgb.LGBMRegressor
    feature_cols: list[str]
    zone: str
    config: TrainingConfig
    cv_metrics: dict
    train_metadata: dict


def save_model_artifact(
    artifact: ModelArtifact,
    output_dir: Path,
) -> Path:
    """
    Serialise ModelArtifact to disk using joblib.

    joblib is preferred over pickle for LightGBM models because:
      - It uses memory-mapped files for large numpy arrays (faster loads)
      - It compresses numpy arrays efficiently by default
      - It is the scikit-learn ecosystem standard

    Output structure:
        output_dir/
          {zone}_lgbm_imbalance.joblib    ← model + feature list + config
          {zone}_train_metadata.json      ← human-readable provenance record

    The .json sidecar allows CI pipelines and dashboards to inspect training
    provenance without loading the binary model file.

    Args:
        artifact:   ModelArtifact dataclass instance.
        output_dir: Directory to write artifacts into.

    Returns:
        Path to the written .joblib file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{artifact.zone}_lgbm_imbalance.joblib"
    joblib.dump(artifact, model_path, compress=("lz4", 3))

    # Human-readable sidecar
    metadata_path = output_dir / f"{artifact.zone}_train_metadata.json"
    sidecar = {
        "zone": artifact.zone,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_cols": artifact.feature_cols,
        "n_features": len(artifact.feature_cols),
        "cv_metrics": artifact.cv_metrics,
        "train_metadata": artifact.train_metadata,
        "lgbm_params": artifact.config.lgbm_params,
        "config": {
            k: v for k, v in asdict(artifact.config).items()
            if k != "lgbm_params"
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=2, default=str)

    logger.info(
        "Model artifact saved → %s (%.1f KB)",
        model_path, model_path.stat().st_size / 1024,
    )
    logger.info("Training metadata saved → %s", metadata_path)
    return model_path


def load_model_artifact(model_path: Path) -> ModelArtifact:
    """
    Reload a serialised ModelArtifact for live inference.

    Usage:
        artifact = load_model_artifact(Path("outputs/models/SE3_lgbm_imbalance.joblib"))
        X_live = df_gold_live[artifact.feature_cols]
        y_pred = artifact.model.predict(X_live)

    Args:
        model_path: Path to the .joblib file written by save_model_artifact().

    Returns:
        ModelArtifact with loaded model, feature list, and metadata.
    """
    artifact: ModelArtifact = joblib.load(model_path)
    logger.info(
        "Model loaded from %s | zone=%s | %d features",
        model_path, artifact.zone, len(artifact.feature_cols),
    )
    return artifact


# ===========================================================================
# Zone-Level Training Orchestrator
# ===========================================================================

def train_zone_model(
    df_gold: pd.DataFrame,
    zone: str,
    config: TrainingConfig,
) -> tuple[ModelArtifact, pd.DataFrame, pd.Series]:
    """
    Full training pipeline for a single bidding zone.

    Executes: data split → CV → final fit → serialise.

    Args:
        df_gold: Full multi-zone Gold layer DataFrame.
        zone:    Target zone (e.g. 'SE3').
        config:  TrainingConfig instance.

    Returns:
        (artifact, X_test, y_test) — the artifact for persistence, and the
        held-out test set for evaluate.py to score.
    """
    logger.info("=" * 60)
    logger.info("Training pipeline | zone=%s", zone)
    logger.info("=" * 60)

    df_zone = df_gold[df_gold["zone"] == zone].copy()
    if df_zone.empty:
        raise ValueError(f"No Gold layer data for zone '{zone}'.")

    feature_cols = resolve_feature_columns(df_zone, config.target_col)

    # Chronological train/test split
    X_train, y_train, X_test, y_test, meta_train, _ = temporal_train_test_split(
        df=df_zone,
        test_hours=config.test_hours,
        target_col=config.target_col,
        feature_cols=feature_cols,
        zone=zone,
    )

    # Walk-forward CV on training window
    cv_metrics = run_cross_validation(X_train, y_train, config, zone)

    # Final production fit
    model = fit_final_model(X_train, y_train, config, zone)

    # Build provenance metadata
    train_metadata = {
        "zone": zone,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_start": str(meta_train["timestamp_utc"].min()),
        "train_end": str(meta_train["timestamp_utc"].max()),
        "n_features": len(feature_cols),
        "best_iteration": int(model.best_iteration_) if model.best_iteration_ else None,
        "target_col": config.target_col,
    }

    artifact = ModelArtifact(
        model=model,
        feature_cols=feature_cols,
        zone=zone,
        config=config,
        cv_metrics=cv_metrics,
        train_metadata=train_metadata,
    )

    # Persist
    save_model_artifact(artifact, Path(config.output_dir))

    return artifact, X_test, y_test


# ===========================================================================
# Multi-Zone Entry Point
# ===========================================================================

def train_all_zones(config: Optional[TrainingConfig] = None) -> dict[str, ModelArtifact]:
    """
    Train independent LightGBM models for every bidding zone in the Gold layer.

    This is the primary entry point for the Model Factory.  It loads the
    Gold layer once and iterates over each zone, producing one serialised
    model artifact per zone.

    Args:
        config: TrainingConfig.  Defaults to TrainingConfig() if None.

    Returns:
        Dict mapping zone label → ModelArtifact for downstream evaluation.
    """
    config = config or TrainingConfig()
    logger.info("Model Factory starting | output_dir=%s", config.output_dir)

    df_gold = load_gold_parquet(
        Path(config.gold_parquet_dir),
        zone_filter=config.zone_filter,
    )

    zones = config.zone_filter or sorted(df_gold["zone"].unique().tolist())
    logger.info("Zones to train: %s", zones)

    artifacts: dict[str, ModelArtifact] = {}
    test_sets: dict[str, tuple[pd.DataFrame, pd.Series]] = {}

    for zone in zones:
        try:
            artifact, X_test, y_test = train_zone_model(df_gold, zone, config)
            artifacts[zone] = artifact
            test_sets[zone] = (X_test, y_test)
        except Exception as exc:
            logger.error(
                "Training failed for zone %s: %s", zone, exc, exc_info=True
            )
            continue

    logger.info(
        "Model Factory complete | %d / %d zones trained successfully",
        len(artifacts), len(zones),
    )

    # Write a combined summary JSON
    summary_path = Path(config.output_dir) / "training_summary.json"
    summary = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "zones": {
            zone: {
                "cv_mean_mae":  a.cv_metrics["mean_mae"],
                "cv_mean_rmse": a.cv_metrics["mean_rmse"],
                "cv_std_mae":   a.cv_metrics["std_mae"],
                "train_rows":   a.train_metadata["train_rows"],
                "test_rows":    a.train_metadata["test_rows"],
                "n_features":   a.train_metadata["n_features"],
            }
            for zone, a in artifacts.items()
        },
    }
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Training summary written → %s", summary_path)

    return artifacts


# ===========================================================================
# CLI Entry Point
# ===========================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Train LightGBM grid imbalance regressors (one per SE zone)."
    )
    parser.add_argument(
        "--gold-dir",
        default="data/gold",
        help="Root directory of the Gold layer Parquet store.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/models",
        help="Directory to write model artifacts and metadata.",
    )
    parser.add_argument(
        "--zones",
        nargs="+",
        default=None,
        help="Bidding zones to train (default: all). E.g. --zones SE1 SE3",
    )
    parser.add_argument(
        "--test-hours",
        type=int,
        default=720,
        help="Hours to reserve as hold-out test set (default: 720 = ~30 days).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of TimeSeriesSplit CV folds (default: 5).",
    )
    args = parser.parse_args()

    cfg = TrainingConfig(
        gold_parquet_dir=args.gold_dir,
        output_dir=args.output_dir,
        zone_filter=args.zones,
        test_hours=args.test_hours,
        n_cv_splits=args.n_splits,
    )

    print("⚠️ Applying Local Test Dataset Configuration Override...")
    
    # Try both ways depending on whether 'cfg' is a dictionary or a custom class/object:
    if isinstance(cfg, dict):
        cfg["test_hours"] = 168
        cfg["min_history_hours"] = 168  # ensure validation matches
    else:
        # If it's an object with attributes:
        setattr(cfg, "test_hours", 168)
        setattr(cfg, "min_history_hours", 168)
        
    train_all_zones(cfg)
