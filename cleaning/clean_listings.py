"""Reproducible HomeMatch cleaning; source facts and modelling estimates stay separate."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_CANDIDATES = [
    'price', 'accommodates', 'bedrooms', 'bathrooms', 'beds',
    'latitude', 'longitude', 'minimum_nights', 'maximum_nights', 'availability_365',
]
CATEGORICAL_CANDIDATES = ['property_type', 'room_type', 'neighbourhood', 'city']


def impute_model_features(data, numeric_method='median'):
    """Impute by original semantic type, before any encoding; never change source facts."""
    if numeric_method not in ('mean', 'median'):
        raise ValueError('numeric_method must be mean or median')
    numeric = pd.DataFrame(index=data.index)
    categorical = pd.DataFrame(index=data.index)
    indicators = pd.DataFrame(index=data.index)
    decisions = []
    for columns, kind, destination in [
        (NUMERIC_CANDIDATES, 'numerical', numeric),
        (CATEGORICAL_CANDIDATES, 'categorical', categorical),
    ]:
        for col in columns:
            if col not in data:
                continue
            observed = data[col].dropna()
            if observed.empty:
                decisions.append({'column': col, 'data_type': kind, 'method': 'exclude: no observed values',
                                  'fill_value': None, 'filled_count': 0})
                continue
            method = numeric_method if kind == 'numerical' else 'mode'
            # A tied mode is resolved deterministically by pandas' sorted first mode.
            fill = float(getattr(observed, method)()) if kind == 'numerical' else observed.mode().iloc[0]
            missing = data[col].isna()
            destination[col] = data[col].fillna(fill)
            if missing.any():
                indicators[col + '_was_missing'] = missing.astype(int)
            decisions.append({'column': col, 'data_type': kind, 'method': method,
                              'fill_value': fill, 'filled_count': int(missing.sum())})
    return numeric, categorical, indicators, decisions


def zscore_filter(numeric, threshold=3.0):
    """One pass after imputation: exclude rows with ANY abs(z) > threshold.

    Population standard deviation (ddof=0). Constant columns have z=0.
    IDs, category codes, missing flags and TF-IDF features never enter this calculation.
    """
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('zscore threshold must be finite and positive')
    means = numeric.mean()
    stds = numeric.std(ddof=0)
    zscores = numeric.sub(means).div(stds.replace(0, np.nan)).fillna(0.0)
    outlier = zscores.abs().gt(threshold).any(axis=1)
    profile = pd.DataFrame({
        'column': numeric.columns, 'mean': means.values, 'population_std': stds.values,
        'flagged_rows': zscores.abs().gt(threshold).sum().values,
    })
    return ~outlier, zscores, profile


def select_numeric_features(numeric, threshold=0.9):
    """Greedy redundancy filter, using abs Pearson correlation after outlier removal.

    Earlier NUMERIC_CANDIDATES have priority; retained features must be nonconstant
    and have abs(r) <= threshold with every previously retained feature.
    This removes redundancy; it does not measure predictive importance.
    """
    if not 0 < threshold <= 1:
        raise ValueError('correlation threshold must be in (0, 1]')
    correlation = numeric.corr(method='pearson')
    keep, decisions = [], []
    for col in numeric:
        record = {'column': col, 'action': 'keep', 'reason': 'nonconstant; no excessive correlation with retained features',
                  'correlated_with': '', 'absolute_correlation': None}
        if numeric[col].nunique(dropna=True) <= 1:
            record.update(action='drop', reason='constant after outlier removal')
        else:
            match = next((other for other in keep if abs(correlation.loc[col, other]) > threshold), None)
            if match is not None:
                record.update(action='drop', reason='redundant above configured correlation threshold',
                              correlated_with=match, absolute_correlation=float(abs(correlation.loc[col, match])))
            else:
                keep.append(col)
        decisions.append(record)
    return keep, correlation, decisions


def clean(input_path, output_dir, threshold=0.5, numeric_method='median',
          zscore_threshold=3.0, correlation_threshold=0.9):
    if not 0 <= threshold <= 1:
        raise ValueError('missing threshold must be between 0 and 1')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(input_path, low_memory=False)
    if source.empty:
        raise ValueError('The input contains no listings')
    if 'id' not in source or source.id.isna().any():
        raise ValueError('Every listing must have a source id')
    counts = [{'stage': 'Input snapshot', 'rows': len(source), 'columns': len(source.columns)}]
    # Full source records, INCLUDING id: similar listings are not duplicates.
    exact = source.duplicated(keep='first')
    source.loc[exact].to_csv(output_dir / 'exact_duplicates_removed.csv', index=False)
    data = source.loc[~exact].copy()
    conflicts = data.id.duplicated(keep=False)
    if conflicts.any():
        data.loc[conflicts].to_csv(output_dir / 'conflicting_ids.csv', index=False)
        raise ValueError('Conflicting records share an id; review conflicting_ids.csv before continuing')
    counts.append({'stage': 'After exact duplicate removal', 'rows': len(data), 'columns': len(data.columns)})
    # Whitespace-only cells count as missing; preserve text case and wording.
    for col in data.select_dtypes(include='object'):
        data[col] = data[col].astype('string').str.strip().replace('', pd.NA)
    profile = pd.DataFrame({'column': data.columns, 'missing_count': data.isna().sum().values,
                            'missing_fraction': data.isna().mean().values})
    dropped = profile.loc[profile.missing_fraction > threshold, 'column'].tolist()
    profile['threshold_decision'] = profile.column.map(lambda c: 'drop' if c in dropped else 'retain')
    profile.to_csv(output_dir / 'missing_values_profile.csv', index=False)
    data = data.drop(columns=dropped)
    counts.append({'stage': 'After missing-column threshold', 'rows': len(data), 'columns': len(data.columns)})
    issues = []
    def flag(col, mask, reason):
        for idx in data.index[mask.fillna(False)]:
            issues.append({'id': int(data.at[idx, 'id']), 'column': col,
                           'source_value': str(data.at[idx, col]), 'reason': reason})
    money = ['price', 'weekly_price', 'monthly_price', 'security_deposit', 'cleaning_fee', 'extra_people']
    numerical = ['latitude', 'longitude', 'accommodates', 'bathrooms', 'bedrooms', 'beds',
                 'guests_included', 'minimum_nights', 'maximum_nights', 'availability_365',
                 'host_listings_count', 'host_total_listings_count', 'host_response_rate']
    for col in money + numerical:
        if col not in data:
            continue
        raw = data[col]
        text = raw.astype('string').str.replace(',', '', regex=False)
        if col in money:
            text = text.str.replace(r'[$€£]', '', regex=True).str.strip()
        if col == 'host_response_rate':
            text = text.str.replace('%', '', regex=False)
        converted = pd.to_numeric(text, errors='coerce')
        flag(col, raw.notna() & converted.isna(), 'unparseable numeric; set to missing')
        data[col] = converted / 100 if col == 'host_response_rate' else converted
        lower, upper = (0, None)
        if col == 'latitude': lower, upper = -90, 90
        elif col == 'longitude': lower, upper = -180, 180
        elif col == 'availability_365': upper = 365
        elif col == 'host_response_rate': upper = 1
        invalid = data[col].notna() & ((data[col] < lower) | ~np.isfinite(data[col]))
        if upper is not None: invalid |= data[col] > upper
        if col in ['price', 'accommodates', 'minimum_nights', 'maximum_nights']:
            invalid |= data[col] == 0
        if col in ['accommodates', 'bedrooms', 'beds', 'guests_included', 'minimum_nights',
                   'maximum_nights', 'availability_365', 'host_listings_count', 'host_total_listings_count']:
            invalid |= data[col].notna() & (data[col] % 1 != 0)
        flag(col, invalid, 'invalid domain value; set to missing, not guessed')
        data.loc[invalid, col] = np.nan
    for col in ['host_has_profile_pic', 'host_identity_verified', 'is_location_exact']:
        if col in data:
            mapped = data[col].astype('string').str.lower().map({'t': 1, 'f': 0, 'true': 1, 'false': 0})
            flag(col, data[col].notna() & mapped.isna(), 'unrecognised boolean; set to missing')
            data[col] = mapped
    if {'minimum_nights', 'maximum_nights'} <= set(data):
        bad_stay = data.maximum_nights < data.minimum_nights
        flag('maximum_nights', bad_stay, 'maximum below minimum; retained for manual review')
    data.to_csv(output_dir / 'listings_cleaned_reviewed.csv', index=False)
    pd.DataFrame(issues, columns=['id','column','source_value','reason']).to_csv(output_dir / 'data_quality_issues.csv', index=False)
    counts.append({'stage': 'Search/display dataset (unknown facts retained as missing)',
                   'rows': len(data), 'columns': len(data.columns)})
    # Imputation and statistical filtering only affect the separate model dataset.
    numeric, categorical, indicators, decisions = impute_model_features(data, numeric_method)
    pd.DataFrame(decisions).to_csv(output_dir / 'model_imputation_log.csv', index=False)
    counts.append({'stage': 'Model fields after imputation (before encoding; excludes id/text)',
                   'rows': len(data), 'columns': len(numeric.columns) + len(categorical.columns)})
    keep_rows, zscores, profile = zscore_filter(numeric, zscore_threshold)
    profile.to_csv(output_dir / 'outlier_profile.csv', index=False)
    zscore_export = zscores.rename(columns=lambda c: 'z_' + c)
    pd.concat([data[['id']], zscore_export], axis=1).to_csv(output_dir / 'z_scores.csv', index=False)
    removed = pd.concat([data.loc[~keep_rows, ['id']], numeric.loc[~keep_rows]], axis=1)
    removed['max_absolute_zscore'] = zscores.loc[~keep_rows].abs().max(axis=1)
    removed['flagged_features'] = zscores.loc[~keep_rows].apply(
        lambda row: ';'.join(row.index[row.abs() > zscore_threshold]), axis=1)
    removed.to_csv(output_dir / 'outliers_removed.csv', index=False)
    if not keep_rows.any():
        raise ValueError('No model rows remain after Z-score filtering')
    model_data = data.loc[keep_rows]
    numeric = numeric.loc[keep_rows]
    categorical = categorical.loc[keep_rows]
    indicators = indicators.loc[keep_rows]
    counts.append({'stage': 'Model fields after one-pass Z-score removal (excludes id/text)',
                   'rows': len(numeric), 'columns': len(numeric.columns) + len(categorical.columns)})
    keep_numeric, correlation, selection = select_numeric_features(numeric, correlation_threshold)
    correlation.to_csv(output_dir / 'correlation_matrix.csv', index_label='feature')
    pd.DataFrame(selection).to_csv(output_dir / 'feature_selection.csv', index=False)
    # Store an interpretable, imputed/selected table before numerical encoding.
    model_table = pd.concat([model_data[['id']], numeric[keep_numeric], categorical], axis=1)
    model_table.to_csv(output_dir / 'model_ready_listings.csv', index=False)
    counts.append({'stage': 'Selected model table (includes source id; excludes text/indicators)',
                   'rows': len(model_table), 'columns': len(model_table.columns)})
    features, feature_names = [], []
    if keep_numeric:
        features.append(sparse.csr_matrix(StandardScaler().fit_transform(numeric[keep_numeric])))
        feature_names.extend(keep_numeric)
    if len(indicators.columns):
        features.append(sparse.csr_matrix(indicators.to_numpy(dtype=float)))
        feature_names.extend(indicators.columns)
    if len(categorical.columns):
        encoder = OneHotEncoder(handle_unknown='ignore')
        features.append(encoder.fit_transform(categorical.astype(str)))
        feature_names.extend(encoder.get_feature_names_out(categorical.columns))
    # One description per listing avoids repeating overlapping title/summary text.
    text_used = False
    if 'description' in model_data:
        # Free text is not a categorical measurement; do not invent a description via mode.
        text = model_data.description.fillna('').astype(str)
        if text.str.contains(r'\w\w', regex=True).sum() >= 2:
            vectorizer = TfidfVectorizer(max_features=500, min_df=2, stop_words='english')
            try:
                tfidf = vectorizer.fit_transform(text)
            except ValueError as exc:
                if 'empty vocabulary' not in str(exc) and 'After pruning' not in str(exc): raise
            else:
                features.append(tfidf)
                feature_names.extend('description_tfidf:' + t for t in vectorizer.get_feature_names_out())
                text_used = True
    if not features:
        raise ValueError('No usable model features remain')
    matrix = sparse.hstack(features, format='csr')
    # Drop constant encoded columns as well; never apply Z-scores to category indicators.
    means = np.asarray(matrix.mean(axis=0)).ravel()
    variances = np.maximum(0, np.asarray(matrix.power(2).mean(axis=0)).ravel() - means ** 2)
    variable = variances > 1e-12
    pd.DataFrame({'feature': feature_names, 'action': np.where(variable, 'keep', 'drop_constant')}).to_csv(
        output_dir / 'encoded_feature_selection.csv', index=False)
    matrix = matrix[:, variable]
    feature_names = np.asarray(feature_names)[variable].tolist()
    if not matrix.shape[1]:
        raise ValueError('All model features are constant; clustering would not be meaningful')
    if not np.isfinite(matrix.data).all():
        raise ValueError('Non-finite model features were produced')
    sparse.save_npz(output_dir / 'clustering_features.npz', matrix)
    model_data[['id']].reset_index(drop=True).to_csv(output_dir / 'clustering_row_ids.csv', index=False)
    pd.DataFrame({'feature': feature_names}).to_csv(output_dir / 'clustering_feature_names.csv', index=False)
    counts.append({'stage': 'Numeric clustering matrix (separate from search data)',
                   'rows': matrix.shape[0], 'columns': matrix.shape[1]})
    pd.DataFrame(counts).to_csv(output_dir / 'stage_counts.csv', index=False)
    audit = {'input_file': Path(input_path).name, 'input_sha256': hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
             'missing_threshold': threshold, 'dropped_columns': dropped, 'exact_duplicates_removed': int(exact.sum()),
             'numeric_imputation': numeric_method, 'categorical_imputation': 'mode before encoding',
             'zscore_threshold': zscore_threshold, 'zscore_ddof': 0, 'zscore_passes': 1,
             'outlier_numeric_columns': list(numeric.columns), 'model_rows_removed_as_outliers': len(removed),
             'correlation_threshold': correlation_threshold, 'retained_numeric_features': keep_numeric,
             'numeric_feature_priority': NUMERIC_CANDIDATES,
             'stages': counts, 'tfidf_included': text_used, 'quality_issue_flags': len(issues),
             'source_currency': 'Unverified: source strings use $; symbol removal is NOT currency conversion',
             'limitations': ['Input is the supplied 54-column snapshot, not a reconstruction of the full Kaggle source.',
                            'Model matrix is exploratory; no clustering accuracy or feature importance has been demonstrated.',
                            'Availability counts do not prove availability on requested dates.',
                            'Unknown operational values remain missing; model estimates must never be used as verified facts.',
                            'Z-score exclusions apply to modelling only; rare legitimate listings may be excluded.',
                            'Correlation filtering reduces redundancy; it does not establish feature importance.',
                            'Feature weighting and scaling require evaluation before deploying a matching model.']}
    (output_dir / 'cleaning_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('cleaning_output'))
    parser.add_argument('--missing-threshold', type=float, default=0.5)
    parser.add_argument('--numeric-imputation', choices=['mean', 'median'], default='median')
    parser.add_argument('--zscore-threshold', type=float, default=3.0)
    parser.add_argument('--correlation-threshold', type=float, default=0.9)
    args = parser.parse_args()
    clean(args.input, args.output, args.missing_threshold, args.numeric_imputation,
          args.zscore_threshold, args.correlation_threshold)
