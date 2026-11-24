import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import sparse

from cleaning.clean_listings import (
    clean, impute_model_features, zscore_filter, select_numeric_features,
)


class CleaningTests(unittest.TestCase):
    def run_cleaning(self, rows, folder):
        source = folder / "input.csv"
        output = folder / "output"
        pd.DataFrame(rows).to_csv(source, index=False)

        with contextlib.redirect_stdout(io.StringIO()):
            clean(source, output)

        return pd.read_csv(output / "listings_cleaned_reviewed.csv")

    def test_preserves_facts_and_removes_only_exact_duplicates(self):
        first = {
            "id": 101,
            "price": "$1,250.50",
            "bedrooms": None,
            "minimum_nights": 365,
            "maximum_nights": 730,
        }
        second = {**first, "id": 102, "bedrooms": 2}

        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_cleaning(
                [first, second, first.copy()],
                Path(temporary),
            )

        # The exact repeat is removed; both distinct listings remain.
        self.assertEqual(result["id"].tolist(), [101, 102])
        self.assertEqual(result.loc[0, "price"], 1250.50)

        # Exactly 50% missing: retain the column and the unknown value.
        self.assertTrue(pd.isna(result.loc[0, "bedrooms"]))
        self.assertEqual(result.loc[1, "bedrooms"], 2)

        # Long booking restrictions must not be shortened.
        self.assertEqual(result["minimum_nights"].tolist(), [365, 365])
        self.assertEqual(result["maximum_nights"].tolist(), [730, 730])

    def test_conflicting_records_with_same_id_are_rejected(self):
        rows = [
            {"id": 101, "price": "$100.00"},
            {"id": 101, "price": "$200.00"},
        ]

        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "Conflicting records"):
                self.run_cleaning(rows, Path(temporary))

    def test_threshold_retains_half_empty_and_drops_over_half(self):
        rows = [
            {'id': i + 1, 'price': 100 + i, 'bedrooms': 1 if i < 2 else None,
             'city': 'Cork' if i == 0 else None}
            for i in range(4)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_cleaning(rows, Path(temporary))
        self.assertIn('bedrooms', result)
        self.assertNotIn('city', result)

    def test_mode_before_encoding_including_numeric_category_labels(self):
        data = pd.DataFrame({'price': [100, 200, 900, None],
                             'property_type': [10, 10, 20, None]})
        numeric, categorical, indicators, log = impute_model_features(data)
        self.assertEqual(numeric.loc[3, 'price'], 200)
        self.assertEqual(categorical.loc[3, 'property_type'], 10)
        self.assertEqual(indicators.loc[3, 'property_type_was_missing'], 1)
        self.assertEqual(next(r['method'] for r in log if r['column'] == 'property_type'), 'mode')
        self.assertTrue(pd.isna(data.loc[3, 'price']))

    def test_mean_option_and_deterministic_mode_tie(self):
        data = pd.DataFrame({'price': [100, 200, 900, None],
                             'city': ['Cork', 'Dublin', None, None]})
        numeric, categorical, _, _ = impute_model_features(data, 'mean')
        self.assertEqual(numeric.loc[3, 'price'], 400)
        self.assertEqual(categorical.loc[2, 'city'], 'Cork')

    def test_zscore_removes_positive_and_negative_extremes(self):
        numeric = pd.DataFrame({'price': [0] * 30 + [100, -100], 'beds': [2] * 32})
        keep, scores, _ = zscore_filter(numeric)
        self.assertEqual(keep.sum(), 30)
        self.assertFalse(keep.iloc[-1])
        self.assertFalse(keep.iloc[-2])
        self.assertTrue((scores['beds'] == 0).all())
        np.testing.assert_allclose(scores['price'], numeric.price / numeric.price.std(ddof=0))

    def test_zscore_boundary_is_strict(self):
        # For [0,0,0,0,10] the last population Z-score equals exactly 2.
        keep, _, _ = zscore_filter(pd.DataFrame({'price': [0, 0, 0, 0, 10]}), threshold=2)
        self.assertTrue(keep.all())

    def test_correlation_reduces_redundancy_and_drops_constants(self):
        numeric = pd.DataFrame({'price': [1, 2, 3, 4, 5],
                                'beds': [2, 4, 6, 8, 10],
                                'bedrooms': [10, 8, 6, 4, 2],
                                'bathrooms': [1, 0, 1, 0, 1],
                                'minimum_nights': [2] * 5})
        keep, corr, decisions = select_numeric_features(numeric)
        self.assertEqual(keep, ['price', 'bathrooms'])
        self.assertAlmostEqual(corr.loc['price', 'bedrooms'], -1)
        self.assertEqual(next(r['correlated_with'] for r in decisions if r['column'] == 'beds'), 'price')

    def test_correlations_compare_against_retained_features_only(self):
        # A-B and B-C exceed .9; A-C does not. C must survive after B is dropped.
        a = np.array([1., -1., 1., -1.])
        b = np.array([1., 1., -1., -1.])
        frame = pd.DataFrame({'a': a, 'b': .95 * a + np.sqrt(1 - .95**2) * b,
                              'c': .85 * a + np.sqrt(1 - .85**2) * b})
        keep, _, _ = select_numeric_features(frame)
        self.assertEqual(keep, ['a', 'c'])

    def test_outlier_filter_keeps_matrix_ids_and_source_data_aligned(self):
        rows = [
            {'id': i + 1, 'price': 100 if i < 20 else 10000,
             'accommodates': 2 + i % 2,
             'city': 'Cork' if i < 19 else 'Dublin'}
            for i in range(21)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = self.run_cleaning(rows, folder)
            output = folder / 'output'
            ids = pd.read_csv(output / 'clustering_row_ids.csv')['id'].tolist()
            removed = pd.read_csv(output / 'outliers_removed.csv')
            model = pd.read_csv(output / 'model_ready_listings.csv')
            matrix = sparse.load_npz(output / 'clustering_features.npz')
            names = pd.read_csv(output / 'clustering_feature_names.csv')
            self.assertEqual(len(source), 21)
            self.assertEqual(ids, list(range(1, 21)))
            self.assertEqual(removed['id'].tolist(), [21])
            self.assertEqual(ids, model['id'].tolist())
            self.assertEqual(matrix.shape, (len(ids), len(names)))
            self.assertFalse(model.isna().any().any())
            self.assertTrue(np.isfinite(matrix.data).all())

    def test_invalid_thresholds_are_rejected(self):
        with self.assertRaises(ValueError):
            zscore_filter(pd.DataFrame({'price': [1, 2]}), 0)
        with self.assertRaises(ValueError):
            select_numeric_features(pd.DataFrame({'price': [1, 2]}), 1.5)


if __name__ == "__main__":
    unittest.main()
