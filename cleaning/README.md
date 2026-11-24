# HomeMatch cleaning review

This package is a revised cleaning pass over the **10,020-row, 54-column `listings_cleaned.csv`** supplied in the repository ZIP. It does not overwrite Andrew's script, the SQLite database or the existing database import format. Review it with the database/search owners before integrating it.

## Run it

Use Python 3.11 or later. From this folder:

```bash
python -m venv .venv
```

On Windows PowerShell, activation is optional; use the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe clean_listings.py --input "C:\path\to\project-main\Data\Cleaned Data\listings_cleaned.csv" --output output
```

On Linux/macOS:

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python clean_listings.py --input /path/to/listings_cleaned.csv --output output
```

The input CSV is already in the repository ZIP and is not duplicated in this package. `--missing-threshold 0.5` is the default: drop columns with **strictly more than 50%** missing, retaining a column at exactly 50%.

## Results from this run

| Stage | Rows | Columns |
|---|---:|---:|
| Supplied snapshot | 10,020 | 54 |
| After exact duplicate removal | 10,020 | 54 |
| Reviewed listing data | 10,020 | 50 |
| Exploratory numeric clustering matrix | 10,020 | 939 |

No full-record duplicates or conflicting listing IDs were found. Different IDs are not merged merely because listings share features. Conflicting records with the same ID stop a future run for manual review.

| Column removed | Missing |
|---|---:|
| notes | 60.56% |
| zipcode | 53.50% |
| weekly_price | 90.42% |
| monthly_price | 93.37% |

Four zero nightly prices were set to missing rather than replaced with invented prices. All listings remain in the output: a priced-search consumer must exclude listings without a valid price. Bedroom counts, minimum stays, maximum stays and annual availability counts are preserved from the supplied snapshot, including six unknown bedroom counts. A zero-bedroom studio is not automatically an error.

The outlier review contains **11,996 field-level IQR flags**, not 11,996 erroneous listings. A listing can have several flags. Values outside an IQR fence are retained for review: expensive properties and long minimum stays can be legitimate. See `outlier_profile.csv` for counts by field and `data_quality_issues.csv` for affected IDs.

## Cleaning decisions and lecturer feedback

* **Duplicates:** compare all original fields including ID before transformations or imputation. Remove only exact repeats. Stop on conflicting versions of an ID.
* **Missingness:** trim whitespace, measure missing cells and export a per-column profile. Apply the adjustable 50% threshold. Unknown facts remain missing in the listing dataset. Median imputation is restricted to the separate modelling copy and is logged; missing indicators distinguish estimates from observations. Missing categories use an explicit `__UNKNOWN__` category rather than pretending the most common location/property type is correct.
* **Prices:** remove formatting symbols and separators and parse numbers. **This does not convert currencies.** The supplied strings use `$`; confirm the source currency before displaying euros, applying euro budgets or estimating monthly costs. Missing fees must not be treated as free.
* **Outliers:** flag with the 1.5-IQR rule. Mark impossible domain values as missing, but do not automatically cap genuine booking restrictions or replace unusual values with averages.
* **Text:** preserve names, descriptions, listing URLs and other retained text for traceability/display. For modelling, encode room type, property type, neighbourhood and city with one-hot encoding. Encode description text with TF-IDF, capped at 500 terms. Description mentions are not verified accessibility or essential-service facts.
* **Column selection:** retain source details in the reviewed listing table; select a separate model feature set based on HomeMatch's requirements, as below. This is an exploratory baseline, not evidence that these are the statistically best features. There is no labelled outcome here for supervised feature selection. PCA/chi-square are not implemented or claimed completed. If a specific assessed analysis is mandatory, agree it with the lecturer and document the target first.

| Model inputs | Reason |
|---|---|
| price | Affordability signal; source currency still needs confirmation |
| accommodates, bedrooms, bathrooms, beds | Capacity and accommodation characteristics |
| latitude, longitude | Location signal; coordinate scaling/weighting requires evaluation |
| minimum_nights, maximum_nights | Booking restrictions; also require explicit eligibility checks in search |
| availability_365 | Historical aggregate availability signal, not requested-date confirmation |
| property_type, room_type, neighbourhood, city | Categorical accommodation/location differences |
| description TF-IDF | Exploratory text similarity; no verified service or amenity claims |

IDs, URLs and host names are excluded from the model matrix. They remain in retained source metadata where present; they are not meaningful distance features. Host popularity/profile characteristics are also excluded from this baseline because the project searches for accommodation suitable for the person, rather than predicting host behaviour.

Numeric model features are standardized after median imputation. Categorical indicators and TF-IDF remain sparse. Block weighting, category consistency and clustering quality still need evaluation before using the matrix in a recommender. If introducing train/test evaluation, fit preprocessing on the training partition only; the included matrix is fitted to this complete snapshot for exploration.

## Which file to use

* `output/listings_cleaned_reviewed.csv`: listing facts and retained display metadata. Unknown factual values remain blank. This has **50 columns** and is not a direct replacement for the existing 12-column `properties_prepared.csv` import.
* `output/clustering_features.npz`: sparse numeric model input. Load with `scipy.sparse.load_npz`. This is not a trained clustering model.
* `output/clustering_row_ids.csv`: listing IDs in exactly the matrix row order.
* `output/clustering_feature_names.csv`: feature names in exactly the matrix column order.
* Remaining CSV/JSON files: cleaning evidence, thresholds, imputation values, outliers and source checksum for the report.

The 54-column input was already processed before this work. This review cannot justify earlier exclusions from the full Kaggle source. The original data remains in the repository archive. Date-specific availability must be checked against calendar rows for the complete requested stay; the annual count is insufficient. Accessibility remains outside the agreed implemented scope. Essential-service proximity requires reliable service coordinates/distances, not keyword detection.

## Verification performed

The full-data run checked source ID order and row preservation; unchanged bedroom, minimum/maximum stay and availability values; conversion of four zero prices to missing; and feature-matrix row alignment. A temporary fixture verified `$1,250.50` parsing, preserving an unknown bedroom value, exact duplicate removal, retaining a below-threshold column, dropping an above-threshold column and refusing conflicting IDs. All checks passed. These checks establish preprocessing behaviour, not property suitability or model quality.

## Suggested report paragraph

We reviewed the supplied 10,020-row, 54-column accommodation dataset. No exact duplicate records were found. Four columns exceeded the 50% missing-value threshold and were removed, leaving 50 columns and 10,020 rows. Currency-formatted fields were converted to numeric values, while four zero nightly prices were marked as unknown. Booking restrictions and bedroom counts were preserved. Potential numerical outliers were flagged for review rather than automatically changed. We retained listing details for search and created a separate 939-feature exploratory clustering matrix using scaled numerical fields, one-hot categories and TF-IDF descriptions. Any model-only imputations were recorded separately. The source currency, date-specific availability and model suitability require further verification.

## GitHub handoff

Create a branch such as `data/cleaning-review`, add this script, requirements and cleaning evidence, and open a pull request for review. Confirm the team's policy before committing generated datasets; they may prefer regenerating outputs locally. Do not overwrite the current database import until field mapping, unknown-value handling and currency are agreed. Keep any later GitHub Actions/Docker work in a separate change.
