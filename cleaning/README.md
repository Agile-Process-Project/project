# HomeMatch cleaning and clustering preparation

This script processes the repository's `Data/Cleaned Data/listings_cleaned.csv` snapshot. It implements the team's clarified missing-value rules, semantic-type imputation, Z-score outlier removal and correlation-based redundancy filtering. It preserves a separate search/display dataset because statistical estimates must not replace verified accommodation facts.

## Run from the repository root

Install `cleaning/requirements.txt` in a Python 3.11+ environment. Then:

```bash
python -m unittest discover -s tests -v
python cleaning/clean_listings.py --input "Data/Cleaned Data/listings_cleaned.csv" --output cleaning_output
```

Options:

| Option | Default | Meaning |
|---|---|---|
| `--missing-threshold` | `0.5` | Drop columns strictly above this missing fraction |
| `--numeric-imputation` | `median` | `median` or `mean`, for numerical model fields |
| `--zscore-threshold` | `3.0` | Remove model rows if any selected numerical field has absolute Z-score strictly above this |
| `--correlation-threshold` | `0.9` | Drop redundant numerical features when absolute Pearson correlation is strictly above this |

## Processing decisions

1. **Duplicates:** remove exact source-record repeats, including ID in the comparison. Stop if different records share an ID. Never merge distinct IDs because they have similar accommodation features.
2. **Missing columns:** trim whitespace-only text to missing, profile missingness after deduplication and remove columns with more than 50% missing values. A column exactly half empty is retained. This threshold operates on the supplied snapshot before parsing/domain checks; any later invalid numeric values are logged and handled as missing.
3. **Conversions:** parse currency-formatted numbers, percentages, counts and booleans. Flag invalid domain values and leave them unknown in the search/display table. Do not overwrite legitimate bedroom counts or booking restrictions. Currency symbols are stripped, not converted: the input uses `$`, and the actual currency remains unverified.
4. **Imputation before encoding:** select the model fields listed below, then fill missing numerical measurements with the median by default. Median is chosen for accommodation measurements that can be skewed, and the team allows mean or median. The mean option is also implemented and tested. Fill missing categorical values with their mode, even if their source labels happen to be numeric. Tied modes use the first sorted mode for reproducibility. Add missingness indicators and log every fill value/count. All-missing model fields after parsing are excluded and logged, rather than assigned an invented value.
5. **Z-score outliers:** calculate `z = (x - mean) / population_standard_deviation` on the ten imputed numerical candidates. Perform **one pass**, dropping a row from modelling when **any absolute Z-score > 3**. Constant columns have Z-score zero. Do not apply this rule to IDs, category codes, missing flags, one-hot indicators or TF-IDF terms. Export all Z-scores and the IDs/reasons of removed rows. The 3 threshold comes from the team's supplied requirements. Statistical extremes can be valid listings; the complete source-fact table is retained separately.
6. **Numerical feature selection:** compute a Pearson correlation matrix **after outlier removal and before scaling**. Remove constant numerical fields. Use a configurable absolute-correlation threshold of **0.90** to remove redundant fields. This threshold is an implementation choice, not a threshold specified in Thomas's notes. Process fields in the priority order below and retain a candidate only if it is not too strongly correlated with any already retained field. Every keep/drop decision is exported. This reduces redundancy; it does not prove feature importance or model accuracy.
7. **Encoding:** standardize the retained numerical fields; one-hot encode categorical fields; encode descriptions with TF-IDF (at most 500 terms). Missing descriptions become empty documents, because free text is not a categorical field to fill with another listing's description. Drop constant encoded columns. Save a sparse numeric matrix with matching row IDs and feature names. Correlation pruning is applied to numerical measurements, not used to impose an ordinal interpretation on categories or perform supervised text selection.

## Model fields and rationale

The numerical priority order is **price, accommodates, bedrooms, bathrooms, beds, latitude, longitude, minimum_nights, maximum_nights, availability_365**. When a redundant pair must be resolved, the earlier field is retained. Affordability and capacity are the primary requirements; accommodation characteristics, location and booking restrictions follow. This deterministic policy should be reviewed if the matching objective changes.

| Fields | Project reason |
|---|---|
| price | Affordability; confirm currency before using budget thresholds |
| accommodates, bedrooms, bathrooms, beds | Capacity and accommodation characteristics |
| latitude, longitude | Location signals; evaluate coordinate weighting before clustering |
| minimum_nights, maximum_nights | Booking restrictions; search must also enforce these explicitly |
| availability_365 | Historical aggregate signal; not confirmation for a requested stay |
| property_type, room_type, neighbourhood, city | Categorical accommodation/location characteristics |
| description | Exploratory text similarity; mentions are not verified services or accessibility |

Identifiers, URLs, host names and unrelated source metadata are not included as model distance features. The source-fact output retains remaining display metadata without imputing unverified facts. The null-filling rule is applied to selected model inputs; it is not a reason to invent listing titles, URLs, dates or source identifiers.

## Verified results on the supplied snapshot

| Stage | Rows | Columns |
|---|---:|---:|
| Input snapshot | 10,020 | 54 |
| Search/display dataset | 10,020 | 50 |
| Imputed model fields before outlier removal (no ID/text/indicators) | 10,020 | 14 |
| Model fields after Z-score removal (no ID/text/indicators) | 9,255 | 14 |
| Selected interpretable model table (includes ID) | 9,255 | 15 |
| Final numeric clustering matrix | 9,255 | 898 |

No exact duplicates were found. The missing-column rule removed `notes`, `zipcode`, `weekly_price` and `monthly_price`. Four zero prices were marked unknown in the source-fact table. The model copy filled four prices, six bedroom counts, seven bathroom counts and 17 bed counts with their medians, and filled 1,482 neighbourhoods and 50 cities with their modes. Imputed neighbourhood/locality values are estimates, not verified property locations.

The Z-score pass removed **765 unique model rows**. Correlation screening retained all ten numerical candidates: the largest absolute pair correlation was approximately **0.835**, below 0.90. No numerical feature was removed solely to force the feature count lower. Category vocabularies change after outlier filtering, and constant encoded columns are also removed; the final matrix has 898 features. These dimensions are run-specific and may change with a different dataset or options.

## Output files

| File | Use |
|---|---|
| `listings_cleaned_reviewed.csv` | 50-column source-fact/search table; unknown values stay missing |
| `model_ready_listings.csv` | Imputed, outlier-filtered, selected model measurements/categories plus ID; not a database import |
| `clustering_features.npz` | Sparse encoded model matrix, load with `scipy.sparse.load_npz` |
| `clustering_row_ids.csv` | IDs in exactly the matrix's row order |
| `clustering_feature_names.csv` | Names in exactly the matrix's column order |
| `missing_values_profile.csv` | Missing counts/fractions and threshold decisions |
| `model_imputation_log.csv` | Semantic types, methods, fill values and counts |
| `z_scores.csv` | Pre-filter model Z-scores for every source ID |
| `outlier_profile.csv` | Z-score mean/std and flag counts per numerical field |
| `outliers_removed.csv` | Excluded model IDs, numerical values, maximum absolute Z-score and triggering fields |
| `correlation_matrix.csv` | Pearson correlations on post-outlier numerical measurements |
| `feature_selection.csv` | Keep/drop decisions for numerical measurements |
| `encoded_feature_selection.csv` | Constant-column removal decisions after encoding |
| `exact_duplicates_removed.csv` | Exact source duplicates removed, if any |
| `data_quality_issues.csv` | Invalid domain/parsing values and inconsistent constraints |
| `stage_counts.csv`, `cleaning_audit.json` | Before/after counts, configuration, source checksum and limitations |

## Tests and integration

Eleven automated tests cover duplicates/conflicting IDs, the exact 50% boundary, categorical mode before encoding (including numeric labels), numerical mean/median, mode ties, positive/negative Z-score extremes, constant columns, strict threshold comparisons, correlation redundancy including negative correlations, retained-feature priority and matrix/ID alignment. A full-data run also verified source facts were preserved, excluded/retained IDs form a complete partition, the model table has no nulls, and all sparse values are finite.

The existing Dockerfile and GitHub Actions workflow run this script and discover the expanded tests without configuration changes. Rebuild Docker after copying the updates. Generated outputs stay ignored by Git and are uploaded as workflow artifacts.

This does not deploy an application, train a clustering algorithm, establish matching quality or replace the existing 12-column database import. Confirm source currency and field mappings before integration. Date-specific availability still requires calendar rows; accessibility and service proximity must not be invented from text. The input already contains only 54 columns, so earlier exclusions from the full Kaggle data cannot be justified by this pass. The complete snapshot is used to fit exploratory preprocessing; fit on training data only if adding train/test evaluation later.

## Report paragraph

We revised the cleaning pipeline to follow the team's clarified rules. Starting with 10,020 rows and 54 columns, we found no exact duplicates and removed four columns with more than 50% missing values. In a separate modelling copy, numerical nulls were filled with medians and categorical nulls with modes before encoding. A single absolute Z-score threshold of 3 removed 765 model rows, leaving 9,255. Pearson correlation screening used a redundancy threshold of 0.90; all ten numerical candidates were retained because the highest absolute pair correlation was approximately 0.835. After scaling, one-hot encoding, TF-IDF and constant-feature removal, the exploratory matrix contained 898 features. IDs, exclusions, fill values, selection decisions and counts were exported for traceability. The complete source-fact table remains separate, and eleven automated tests passed.
