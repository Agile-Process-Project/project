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


def clean(input_path, output_dir, threshold=0.5):
    if not 0 <= threshold <= 1:
        raise ValueError('missing threshold must be between 0 and 1')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(input_path, low_memory=False)
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
    bad_stay = data.maximum_nights < data.minimum_nights
    flag('maximum_nights', bad_stay, 'maximum below minimum; retained for manual review')
    # IQR flags are review signals, not proof of error. Do not change stay constraints.
    outlier_report = []
    for col in money + numerical:
        if col not in data: continue
        s = data[col].dropna()
        if s.empty: continue
        q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (data[col] < low) | (data[col] > high)
        outlier_report.append({'column': col, 'iqr_lower': low, 'iqr_upper': high,
                               'flagged_count': int(mask.sum()), 'action': 'retain; review against source'})
        flag(col, mask, 'outside 1.5 IQR; retained, not automatically an error')
    data.to_csv(output_dir / 'listings_cleaned_reviewed.csv', index=False)
    pd.DataFrame(issues, columns=['id','column','source_value','reason']).to_csv(output_dir / 'data_quality_issues.csv', index=False)
    pd.DataFrame(outlier_report).to_csv(output_dir / 'outlier_profile.csv', index=False)
    counts.append({'stage': 'Search/display dataset (unknown facts retained as missing)',
                   'rows': len(data), 'columns': len(data.columns)})
    # Modelling copy only: every imputation is logged and gets a missing indicator.
    features, feature_names, decisions = [], [], []
    numeric_candidates = ['price', 'accommodates', 'bedrooms', 'bathrooms', 'beds',
                          'latitude', 'longitude', 'minimum_nights', 'maximum_nights', 'availability_365']
    numeric = pd.DataFrame(index=data.index)
    for col in numeric_candidates:
        if col not in data or data[col].notna().sum() == 0: continue
        med = float(data[col].median())
        numeric[col] = data[col].fillna(med)
        if data[col].isna().any(): numeric[col + '_was_missing'] = data[col].isna().astype(int)
        decisions.append({'column': col, 'method': 'median in model copy only',
                          'fill_value': med, 'filled_count': int(data[col].isna().sum())})
    features.append(sparse.csr_matrix(StandardScaler().fit_transform(numeric)))
    feature_names.extend(numeric.columns)
    categories = [c for c in ['property_type', 'room_type', 'neighbourhood', 'city'] if c in data]
    if categories:
        categorical = data[categories].fillna('__UNKNOWN__').astype(str)
        encoder = OneHotEncoder(handle_unknown='ignore')
        features.append(encoder.fit_transform(categorical))
        feature_names.extend(encoder.get_feature_names_out(categories))
    # One description per listing avoids repeating overlapping title/summary text.
    text_used = False
    if 'description' in data:
        text = data.description.fillna('').astype(str)
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
    matrix = sparse.hstack(features, format='csr')
    sparse.save_npz(output_dir / 'clustering_features.npz', matrix)
    data[['id']].reset_index(drop=True).to_csv(output_dir / 'clustering_row_ids.csv', index=False)
    pd.DataFrame({'feature': feature_names}).to_csv(output_dir / 'clustering_feature_names.csv', index=False)
    pd.DataFrame(decisions).to_csv(output_dir / 'model_imputation_log.csv', index=False)
    counts.append({'stage': 'Numeric clustering matrix (separate from search data)',
                   'rows': matrix.shape[0], 'columns': matrix.shape[1]})
    pd.DataFrame(counts).to_csv(output_dir / 'stage_counts.csv', index=False)
    audit = {'input_file': Path(input_path).name, 'input_sha256': hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
             'missing_threshold': threshold, 'dropped_columns': dropped, 'exact_duplicates_removed': int(exact.sum()),
             'stages': counts, 'tfidf_included': text_used, 'quality_issue_flags': len(issues),
             'source_currency': 'Unverified: source strings use $; symbol removal is NOT currency conversion',
             'limitations': ['Input is the supplied 54-column snapshot, not a reconstruction of the full Kaggle source.',
                            'Model matrix is exploratory; no clustering accuracy or feature importance has been demonstrated.',
                            'Availability counts do not prove availability on requested dates.',
                            'Unknown operational values remain missing; model estimates must never be used as verified facts.',
                            'Feature weighting and scaling require evaluation before deploying a matching model.']}
    (output_dir / 'cleaning_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('cleaning_output'))
    parser.add_argument('--missing-threshold', type=float, default=0.5)
    args = parser.parse_args()
    clean(args.input, args.output, args.missing_threshold)
