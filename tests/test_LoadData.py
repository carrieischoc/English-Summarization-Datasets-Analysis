import unittest
from LoadData import load_data


class TestStats(unittest.TestCase):
    def test_LoadData(self):
        dataset = load_data("wiki_lingua")
        # the original row of index 14 is empty
        dataset = dataset.select([0, 14])
        expected_num_rows = 2
        expected_features = ["section_name", "source", "target"]
        self.assertEqual(expected_num_rows, dataset.num_rows)
        self.assertEqual(expected_features, dataset.column_names)


if __name__ == "__main__":
    unittest.main()
