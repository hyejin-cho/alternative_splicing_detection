# coding: UTF-8
#!/usr/bin/env python
"""
filter_AS_artifacts.py
======================
Train a gradient boosting classifier to distinguish true alternative
splicing events from alignment artifacts, then apply it to filter your
detect_AS.py output files.

Two modes
---------
  TRAIN mode  : train a new model from final_training_set.tsv and save it
  PREDICT mode: load a saved model and apply it to new events files
  BOTH        : train then immediately apply (default)

Usage
-----
    # Train + apply in one step (recommended first run)
    python filter_AS_artifacts.py \\
        --training    final_training_set.tsv \\
        --events-dir  output_dir/ \\
        --output-dir  filtered_output/ \\
        --model-out   artifact_filter.pkl

    # Train only (save model for later)
    python filter_AS_artifacts.py \\
        --mode        train \\
        --training    final_training_set.tsv \\
        --model-out   artifact_filter.pkl

    # Predict only (use saved model)
    python filter_AS_artifacts.py \\
        --mode        predict \\
        --model-in    artifact_filter.pkl \\
        --events-dir  output_dir/ \\
        --output-dir  filtered_output/

Features used
-------------
  From events files (always present):
    coverage, exclusion1, exclusion2, inclusion, over_boundaries, PSI
    Derived: total_reads, exc_ratio, psi_bin, intron_length

  From splice site annotator (present when --fasta was used):
    is_canonical, splice_site_score, is_known_noncanonical,
    maxentscan_donor_score, maxentscan_acceptor_score,
    is_annotated_junction, donor_in_gtf, acceptor_in_gtf,
    both_sites_in_gtf, dist_to_nearest_donor, dist_to_nearest_acceptor

  Categorical (label-encoded):
    as_type (ES / A3SS / A5SS / IR / MXE), group (group1 / group2)

Output
------
  For each input events file:
    filtered_output/original_filename         kept events (label=1)
    filtered_output/original_filename.removed removed artifacts (label=0)
  Summary:
    filtered_output/filter_summary.tsv
    artifact_filter_model_report.txt

Dependencies
------------
    pip install pandas numpy scikit-learn lightgbm shap joblib
    (XGBoost used as fallback if LightGBM unavailable)
"""

import os
import sys
import glob
import argparse
import logging
import warnings
import pickle
from datetime import datetime as dt

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             classification_report, precision_recall_curve)
from sklearn.preprocessing import LabelEncoder
import joblib

# --- Gradient boosting library (LightGBM preferred, XGBoost fallback) -------
try:
    import lightgbm as lgb
    _GB_LIB = 'lightgbm'
except ImportError:
    try:
        from xgboost import XGBClassifier
        _GB_LIB = 'xgboost'
    except ImportError:
        print('ERROR: install lightgbm or xgboost:\n'
              '  pip install lightgbm\n  pip install xgboost')
        sys.exit(1)

# --- SHAP (optional) --------------------------------------------------------
try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    warnings.warn('shap not installed — feature importance plot skipped.\n'
                  'Install with: pip install shap', ImportWarning)


# ===========================================================================
# Constants
# ===========================================================================

# Base feature columns produced by detect_AS.py _writeEvent()
_BASE_FEATURES = [
    'coverage', 'exclusion1', 'exclusion2',
    'inclusion', 'over_boundaries', 'PSI'
]

# Splice site feature columns from splice_site_features.py
_SPLICE_FEATURES = [
    'is_canonical', 'splice_site_score', 'is_known_noncanonical',
    'maxentscan_donor_score', 'maxentscan_acceptor_score',
    'is_annotated_junction', 'donor_in_gtf', 'acceptor_in_gtf',
    'both_sites_in_gtf', 'dist_to_nearest_donor', 'dist_to_nearest_acceptor'
]

# Categorical columns to label-encode
_CAT_COLS = ['as_type', 'group']

# ID columns (never used as features)
_ID_COLS = ['gene::tx', 'id', 'source_file', 'chrom_resolved',
            'strand_resolved', 'source', 'label',
            'chrom', 'intron_start', 'intron_end', 'strand',
            'gene_id', 'transcript_id', 'reason']


# ===========================================================================
# Feature engineering
# ===========================================================================

def _parse_id_coords(id_val: str) -> tuple:
    """
    Parse intron_start and intron_end from the 'id' column.
    MXE format: start1_end1Nstart2_end2 -> use first component.
    Returns (start, end) or (np.nan, np.nan) on failure.
    """
    try:
        first = str(id_val).split('N')[0]
        parts = first.strip().split('_')
        if len(parts) >= 2:
            return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        pass
    return np.nan, np.nan


def engineer_features(df: pd.DataFrame,
                       filename: str = '',
                       as_type: str = '') -> pd.DataFrame:
    """
    Add derived features and encode categoricals on a copy of df.

    New features added
    ------------------
    intron_start, intron_end  : parsed from 'id' column
    intron_length             : intron_end - intron_start
    total_reads               : coverage + exclusion1 + exclusion2 + inclusion
    exc_ratio                 : (exclusion1 + exclusion2) / (total_reads + 1)
    inc_ratio                 : inclusion / (total_reads + 1)
    psi_bin                   : PSI rounded to 0.1 bins (0.0 to 1.0)
    has_coverage              : 1 if coverage > 0 else 0
    as_type_enc               : label-encoded AS type
    group_enc                 : label-encoded group (group1/group2)
    """
    df = df.copy()

    # --- Parse coordinates from id ----------------------------------------
    if 'id' in df.columns:
        coords = df['id'].apply(
            lambda x: pd.Series(_parse_id_coords(x),
                                 index=['intron_start', 'intron_end']))
        df['intron_start'] = coords['intron_start']
        df['intron_end']   = coords['intron_end']
    else:
        df['intron_start'] = np.nan
        df['intron_end']   = np.nan

    # --- Intron length -------------------------------------------------------
    df['intron_length'] = (df['intron_end'] - df['intron_start']).clip(lower=0)

    # --- Numeric safety: coerce non-numeric base features to NaN ------------
    for col in _BASE_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0.0

    # --- Derived count features ----------------------------------------------
    exc1 = df.get('exclusion1', pd.Series(0, index=df.index))
    exc2 = df.get('exclusion2', pd.Series(0, index=df.index))
    inc  = df.get('inclusion',  pd.Series(0, index=df.index))
    cov  = df.get('coverage',   pd.Series(0, index=df.index))

    # Handle non-numeric exclusion values (some events store '_')
    exc1 = pd.to_numeric(exc1, errors='coerce').fillna(0)
    exc2 = pd.to_numeric(exc2, errors='coerce').fillna(0)
    inc  = pd.to_numeric(inc,  errors='coerce').fillna(0)
    cov  = pd.to_numeric(cov,  errors='coerce').fillna(0)

    total = cov + exc1 + exc2 + inc
    df['total_reads']  = total
    df['exc_ratio']    = (exc1 + exc2) / (total + 1)
    df['inc_ratio']    = inc           / (total + 1)
    df['psi_bin']      = (df['PSI'].clip(0, 1) * 10).round() / 10
    df['has_coverage'] = (cov > 0).astype(int)

    # --- Splice site features (coerce, fill missing with -1) ----------------
    for col in _SPLICE_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1)

    # --- AS type (from filename or passed argument) --------------------------
    if 'as_type' not in df.columns:
        if as_type:
            df['as_type'] = as_type
        elif filename:
            # Extract from filename pattern: prefix_group_TYPE_sample_events.txt
            name = os.path.basename(filename).replace('_events.txt', '')
            parts = name.split('_')
            # AS type is uppercase among the parts
            as_types = {'ES', 'A3SS', 'A5SS', 'IR', 'MXE', 'RI'}
            found = next((p.upper() for p in parts
                          if p.upper() in as_types), 'UNKNOWN')
            df['as_type'] = found
        else:
            df['as_type'] = 'UNKNOWN'

    # --- Group (from filename) -----------------------------------------------
    if 'group' not in df.columns:
        if filename:
            name = os.path.basename(filename).lower()
            if 'group1' in name:
                df['group'] = 'group1'
            elif 'group2' in name:
                df['group'] = 'group2'
            else:
                df['group'] = 'unknown'
        else:
            df['group'] = 'unknown'

    # --- Label-encode categoricals ------------------------------------------
    for cat_col in _CAT_COLS:
        enc_col = cat_col + '_enc'
        if cat_col in df.columns:
            le = LabelEncoder()
            df[enc_col] = le.fit_transform(
                df[cat_col].fillna('unknown').astype(str))
        else:
            df[enc_col] = 0

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Return the list of feature columns to use for training / prediction.
    Includes base features, derived features, splice features (if present),
    and encoded categoricals. Excludes ID and label columns.
    """
    derived = ['intron_length', 'total_reads', 'exc_ratio',
               'inc_ratio', 'psi_bin', 'has_coverage',
               'as_type_enc', 'group_enc']

    candidates = (_BASE_FEATURES + derived +
                  _SPLICE_FEATURES + ['intron_start', 'intron_end'])

    # Keep only columns that exist in the DataFrame
    feature_cols = [c for c in candidates if c in df.columns]

    # Remove any ID/label columns that slipped through
    feature_cols = [c for c in feature_cols if c not in _ID_COLS]

    logging.info('Feature columns (%d): %s', len(feature_cols), feature_cols)
    return feature_cols


# ===========================================================================
# Training data loader
# ===========================================================================

def load_training_data(training_file: str) -> tuple:
    """
    Load final_training_set.tsv, apply feature engineering, and return
    (X, y, feature_columns, full_df).
    """
    logging.info('Loading training data: %s', training_file)
    df = pd.read_csv(training_file, sep='\t', low_memory=False)

    if 'label' not in df.columns:
        raise ValueError(
            f'"label" column not found in {training_file}.\n'
            f'Columns present: {list(df.columns)}'
        )

    print(f'\nTraining data loaded: {len(df):,} rows')
    print(f'  Positives (label=1): {(df.label==1).sum():,}')
    print(f'  Negatives (label=0): {(df.label==0).sum():,}')
    print(f'  Class ratio (neg:pos): '
          f'{(df.label==0).sum() / max((df.label==1).sum(), 1):.2f}')

    # Determine AS type from source_file if available
    if 'source_file' in df.columns:
        as_types = {'ES', 'A3SS', 'A5SS', 'IR', 'MXE', 'RI'}
        def _extract_as_type(src):
            if pd.isna(src):
                return 'UNKNOWN'
            parts = str(src).replace('_events.txt', '').split('_')
            return next((p.upper() for p in parts
                         if p.upper() in as_types), 'UNKNOWN')
        df['as_type'] = df['source_file'].apply(_extract_as_type)

    df = engineer_features(df, filename='')
    y  = df['label'].astype(int).values

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values

    print(f'  Feature columns ({len(feature_cols)}): {feature_cols}')
    return X, y, feature_cols, df


# ===========================================================================
# Model builder
# ===========================================================================

def build_model(class_ratio: float):
    """
    Build a LightGBM or XGBoost classifier appropriate for this dataset size.
    class_ratio = n_negative / n_positive
    """
    if _GB_LIB == 'lightgbm':
        model = lgb.LGBMClassifier(
            n_estimators      = 500,
            num_leaves        = 31,        # conservative for small datasets
            max_depth         = -1,
            learning_rate     = 0.05,
            min_child_samples = 20,        # prevents overfitting on small data
            subsample         = 0.8,
            colsample_bytree  = 0.8,
            reg_alpha         = 0.1,
            reg_lambda        = 1.0,
            is_unbalance      = True,      # handles class imbalance natively
            random_state      = 42,
            n_jobs            = -1,
            verbose           = -1,
        )
        print(f'Model: LightGBM  (class_ratio={class_ratio:.2f})')
    else:
        scale = class_ratio if class_ratio > 1 else 1.0
        model = XGBClassifier(
            n_estimators      = 500,
            max_depth         = 5,
            learning_rate     = 0.05,
            subsample         = 0.8,
            colsample_bytree  = 0.8,
            reg_alpha         = 0.1,
            reg_lambda        = 1.0,
            scale_pos_weight  = scale,
            use_label_encoder = False,
            eval_metric       = 'aucpr',
            random_state      = 42,
            n_jobs            = -1,
        )
        print(f'Model: XGBoost  (scale_pos_weight={scale:.2f})')
    return model


# ===========================================================================
# Training and evaluation
# ===========================================================================

def train_model(X: np.ndarray, y: np.ndarray,
                feature_cols: list, n_folds: int = 5) -> tuple:
    """
    Train with stratified k-fold cross-validation and fit a final model
    on the full training set.

    Returns (final_model, cv_results_dict)
    """
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    class_ratio = n_neg / max(n_pos, 1)

    model = build_model(class_ratio)

    # --- Cross-validation ----------------------------------------------------
    print(f'\n--- {n_folds}-fold cross-validation ---')
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Collect fold-level PR-AUC and ROC-AUC
    pr_aucs  = []
    roc_aucs = []

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        fold_model = build_model(class_ratio)
        fold_model.fit(X_tr, y_tr)
        proba = fold_model.predict_proba(X_val)[:, 1]

        pr_auc  = average_precision_score(y_val, proba)
        roc_auc = roc_auc_score(y_val, proba)
        pr_aucs.append(pr_auc)
        roc_aucs.append(roc_auc)
        print(f'  Fold {fold}: PR-AUC={pr_auc:.4f}  ROC-AUC={roc_auc:.4f}')

    print(f'  Mean PR-AUC : {np.mean(pr_aucs):.4f} ± {np.std(pr_aucs):.4f}')
    print(f'  Mean ROC-AUC: {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}')

    cv_results = {
        'pr_auc_mean':  float(np.mean(pr_aucs)),
        'pr_auc_std':   float(np.std(pr_aucs)),
        'roc_auc_mean': float(np.mean(roc_aucs)),
        'roc_auc_std':  float(np.std(roc_aucs)),
        'n_folds':      n_folds,
    }

    # --- Final model on full training set ------------------------------------
    print('\nFitting final model on full training set ...')
    final_model = build_model(class_ratio)
    final_model.fit(X, y)
    print('Done.')

    # --- Final model evaluation on training set (optimistic — for reporting) -
    train_proba = final_model.predict_proba(X)[:, 1]
    train_pr    = average_precision_score(y, train_proba)
    print(f'Training PR-AUC (full set): {train_pr:.4f}')

    return final_model, cv_results


# ===========================================================================
# Feature importance via SHAP
# ===========================================================================

def compute_shap_importance(model, X: np.ndarray,
                             feature_cols: list,
                             output_dir: str):
    """
    Compute SHAP values and save a feature importance TSV.
    Only runs if shap is installed.
    """
    if not _SHAP_AVAILABLE:
        return

    print('\nComputing SHAP feature importance ...')
    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # For binary classification shap_values may be a list [neg, pos]
        if isinstance(shap_values, list):
            shap_arr = shap_values[1]
        else:
            shap_arr = shap_values

        mean_abs_shap = np.abs(shap_arr).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature':         feature_cols,
            'mean_abs_shap':   mean_abs_shap,
        }).sort_values('mean_abs_shap', ascending=False)

        shap_file = os.path.join(output_dir, 'shap_importance.tsv')
        importance_df.to_csv(shap_file, sep='\t', index=False)

        print('Top 10 features by SHAP:')
        for _, row in importance_df.head(10).iterrows():
            print(f'  {row.feature:<35}: {row.mean_abs_shap:.4f}')
        print(f'SHAP importance saved: {shap_file}')

    except Exception as exc:
        logging.warning('SHAP computation failed: %s', exc)
        print(f'  SHAP skipped: {exc}')


# ===========================================================================
# Model persistence
# ===========================================================================

def save_model(model, feature_cols: list,
               cv_results: dict, model_path: str):
    """Save model + metadata as a single pickle."""
    bundle = {
        'model':        model,
        'feature_cols': feature_cols,
        'cv_results':   cv_results,
        'gb_lib':       _GB_LIB,
        'saved_at':     dt.now().isoformat(),
    }
    joblib.dump(bundle, model_path)
    print(f'Model saved: {model_path}')


def load_model(model_path: str) -> tuple:
    """Load model bundle and return (model, feature_cols, cv_results)."""
    bundle = joblib.load(model_path)
    print(f'Model loaded: {model_path}')
    print(f'  Saved at    : {bundle.get("saved_at", "unknown")}')
    print(f'  GB library  : {bundle.get("gb_lib", "unknown")}')
    cv = bundle.get('cv_results', {})
    if cv:
        print(f'  CV PR-AUC   : {cv.get("pr_auc_mean", 0):.4f} '
              f'± {cv.get("pr_auc_std", 0):.4f}')
    return bundle['model'], bundle['feature_cols'], cv


# ===========================================================================
# Prediction and filtering
# ===========================================================================

def predict_events_file(events_file: str,
                         model,
                         feature_cols: list,
                         threshold: float,
                         output_dir: str) -> dict:
    """
    Load one events file, predict artifact probability for each junction,
    and write two output files:
      kept    : junctions predicted as true (label=1, prob >= threshold)
      removed : junctions predicted as artifacts (label=0, prob < threshold)

    Returns a summary dict.
    """
    df = pd.read_csv(events_file, sep='\t', low_memory=False)
    if df.empty:
        logging.warning('Empty file skipped: %s', events_file)
        return {}

    n_total = len(df)
    df_feat = engineer_features(df, filename=events_file)

    # Align feature columns — fill any missing with 0
    X_pred = np.zeros((len(df_feat), len(feature_cols)), dtype=float)
    for j, col in enumerate(feature_cols):
        if col in df_feat.columns:
            X_pred[:, j] = pd.to_numeric(
                df_feat[col], errors='coerce').fillna(0).values

    # Predict
    proba   = model.predict_proba(X_pred)[:, 1]
    predict = (proba >= threshold).astype(int)

    # Attach predictions to original DataFrame
    df['artifact_prob']     = proba.round(4)
    df['predicted_label']   = predict

    # Split into kept and removed
    kept    = df[df['predicted_label'] == 1].drop(columns=['predicted_label'])
    removed = df[df['predicted_label'] == 0].drop(columns=['predicted_label'])

    # Write output files
    os.makedirs(output_dir, exist_ok=True)
    base_nm      = os.path.basename(events_file)
    kept_path    = os.path.join(output_dir, base_nm)
    removed_path = os.path.join(output_dir, base_nm + '.removed')

    kept.to_csv(kept_path,    sep='\t', index=False)
    removed.to_csv(removed_path, sep='\t', index=False)

    n_kept    = len(kept)
    n_removed = len(removed)
    pct_kept  = 100 * n_kept / max(n_total, 1)

    print(f'  {base_nm}')
    print(f'    Total: {n_total:,}  '
          f'Kept: {n_kept:,} ({pct_kept:.1f}%)  '
          f'Removed: {n_removed:,}')

    return {
        'file':      base_nm,
        'n_total':   n_total,
        'n_kept':    n_kept,
        'n_removed': n_removed,
        'pct_kept':  round(pct_kept, 2),
    }


# ===========================================================================
# Model report
# ===========================================================================

def write_model_report(cv_results: dict,
                        feature_cols: list,
                        summary_rows: list,
                        threshold: float,
                        report_path: str):
    lines = []
    sep = '=' * 60
    lines += [sep, 'ARTIFACT FILTER MODEL REPORT',
              f'Generated : {dt.now().strftime("%Y-%m-%d %H:%M:%S")}',
              f'GB library: {_GB_LIB}',
              sep, '']

    lines.append('--- Cross-validation performance ---')
    lines.append(f'  Folds            : {cv_results.get("n_folds", "N/A")}')
    lines.append(f'  PR-AUC  (mean±sd): '
                 f'{cv_results.get("pr_auc_mean", 0):.4f} '
                 f'± {cv_results.get("pr_auc_std", 0):.4f}')
    lines.append(f'  ROC-AUC (mean±sd): '
                 f'{cv_results.get("roc_auc_mean", 0):.4f} '
                 f'± {cv_results.get("roc_auc_std", 0):.4f}')
    lines.append('')

    lines.append(f'--- Features used ({len(feature_cols)}) ---')
    for fc in feature_cols:
        lines.append(f'  {fc}')
    lines.append('')

    lines.append(f'--- Prediction threshold: {threshold} ---')
    lines.append('  Junctions with artifact_prob >= threshold -> KEPT')
    lines.append('  Junctions with artifact_prob <  threshold -> REMOVED')
    lines.append('')

    if summary_rows:
        lines.append('--- Per-file filtering summary ---')
        total_in  = sum(r['n_total']   for r in summary_rows)
        total_out = sum(r['n_kept']    for r in summary_rows)
        total_rem = sum(r['n_removed'] for r in summary_rows)
        lines.append(f'  {"File":<45} {"Total":>7} {"Kept":>7} {"Removed":>8}')
        lines.append('  ' + '-' * 70)
        for r in summary_rows:
            lines.append(
                f'  {r["file"]:<45} {r["n_total"]:>7,} '
                f'{r["n_kept"]:>7,} {r["n_removed"]:>8,}')
        lines.append('  ' + '-' * 70)
        lines.append(
            f'  {"TOTAL":<45} {total_in:>7,} '
            f'{total_out:>7,} {total_rem:>8,}')
        overall_pct = 100 * total_out / max(total_in, 1)
        lines.append(f'\n  Overall retention rate: {overall_pct:.1f}%')
    lines.append('')
    lines.append(sep)

    report = '\n'.join(lines)
    with open(report_path, 'w') as fh:
        fh.write(report)
    print(f'\nModel report saved: {report_path}')
    print('\n' + report)


# ===========================================================================
# Main pipeline
# ===========================================================================

def run_pipeline(args):
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir,
                               'artifact_filter_model_report.txt')
    model        = None
    feature_cols = None
    cv_results   = {}

    # ── TRAIN ────────────────────────────────────────────────────────────────
    if args.mode in ('train', 'both'):
        if not args.training:
            print('ERROR: --training is required for train mode')
            sys.exit(1)

        X, y, feature_cols, train_df = load_training_data(args.training)
        model, cv_results = train_model(X, y, feature_cols, args.cv_folds)

        # SHAP feature importance
        compute_shap_importance(
            model, X, feature_cols, args.output_dir)

        # Save model
        model_out = args.model_out or os.path.join(
            args.output_dir, 'artifact_filter.pkl')
        save_model(model, feature_cols, cv_results, model_out)

    # ── LOAD ─────────────────────────────────────────────────────────────────
    if args.mode in ('predict', 'both') and model is None:
        if not args.model_in:
            print('ERROR: --model-in is required for predict mode')
            sys.exit(1)
        model, feature_cols, cv_results = load_model(args.model_in)

    # ── PREDICT ──────────────────────────────────────────────────────────────
    summary_rows = []
    if args.mode in ('predict', 'both'):
        # Collect events files
        events_files = []
        if args.events:
            for pattern in args.events:
                events_files.extend(glob.glob(pattern))
        if args.events_dir:
            for d in args.events_dir:
                events_files.extend(
                    glob.glob(os.path.join(d, '*_events.txt')))

        # Deduplicate
        events_files = list(dict.fromkeys(events_files))

        if not events_files:
            print('WARNING: no events files found for prediction.')
        else:
            print(f'\n--- Filtering {len(events_files)} events files '
                  f'(threshold={args.threshold}) ---')
            for ef in sorted(events_files):
                row = predict_events_file(
                    ef, model, feature_cols,
                    args.threshold, args.output_dir)
                if row:
                    summary_rows.append(row)

            # Write summary TSV
            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)
                summary_path = os.path.join(
                    args.output_dir, 'filter_summary.tsv')
                summary_df.to_csv(summary_path, sep='\t', index=False)
                print(f'\nFilter summary: {summary_path}')

    # ── REPORT ───────────────────────────────────────────────────────────────
    write_model_report(
        cv_results, feature_cols or [],
        summary_rows, args.threshold, report_path)


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Train a gradient boosting artifact filter and/or apply it\n'
            'to detect_AS.py output files.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--mode', choices=['train', 'predict', 'both'],
        default='both',
        help='train: fit and save model only\n'
             'predict: load saved model and filter events files\n'
             'both: train then apply (default)')

    train_grp = parser.add_argument_group('Training')
    train_grp.add_argument('--training', metavar='FILE',
        help='final_training_set.tsv from build_training_set.py')
    train_grp.add_argument('--cv-folds', metavar='INT', type=int, default=5,
        help='Cross-validation folds (default: 5)')
    train_grp.add_argument('--model-out', metavar='FILE',
        default=None,
        help='Path to save trained model (default: output_dir/artifact_filter.pkl)')

    pred_grp = parser.add_argument_group('Prediction')
    pred_grp.add_argument('--model-in', metavar='FILE', default=None,
        help='Path to saved model file (required for predict mode)')
    pred_grp.add_argument('--events', metavar='PATTERN', nargs='+',
        default=None,
        help='Events files or glob patterns to filter')
    pred_grp.add_argument('--events-dir', metavar='DIR', nargs='+',
        default=None,
        help='Directories to search for *_events.txt files')
    pred_grp.add_argument('--threshold', metavar='FLOAT', type=float,
        default=0.5,
        help='Probability threshold for keeping a junction (default: 0.5)\n'
             'Lower = more permissive (keep more junctions)\n'
             'Higher = more strict (remove more junctions)')
    pred_grp.add_argument('--output-dir', metavar='DIR',
        default='filtered_output',
        help='Output directory for filtered files (default: filtered_output)')

    parser.add_argument('--log', metavar='FILE',
        default='filter_AS_artifacts.log')

    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG, filemode='w',
        format='[%(levelname)s] %(asctime)s %(message)s',
        filename=args.log
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    logging.getLogger().addHandler(console)

    print(f'Start Time  = {dt.now().strftime("%H:%M:%S")}')
    print(f'Mode        = {args.mode}')
    print(f'GB library  = {_GB_LIB}')
    print(f'SHAP        = {"available" if _SHAP_AVAILABLE else "not installed"}')

    run_pipeline(args)

    print(f'\nEnd Time = {dt.now().strftime("%H:%M:%S")}')


if __name__ == '__main__':
    main()
