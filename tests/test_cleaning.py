import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cleaning.clean_listings import clean


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


if __name__ == "__main__":
    unittest.main()