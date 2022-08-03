import unittest
import pandas as pd
from datasets import Dataset
from inspection import lens_cal
from similarity import compute_similarity


class TestStats(unittest.TestCase):

    article_text = ["This is a test.", "This is another better test."]
    summary_text = ["This is another better test."]
    data_dic = {"source": [article_text], "target": [summary_text]}
    data_df = pd.DataFrame(data_dic)
    dataset = Dataset.from_pandas(data_df)

    def test_length(self):
        length_ws = lens_cal(self.dataset)
        length_spacy = lens_cal(self.dataset, "spacy")
        expected_lens_ws_src_mean = 9.0
        expected_lens_ws_tg_mean = 5.0
        expected_lens_spacy_src_mean = 11.0
        expected_lens_spacy_tg_mean = 6.0
        self.assertEqual(expected_lens_ws_src_mean, length_ws.src.mean)
        self.assertEqual(expected_lens_ws_tg_mean, length_ws.tg.mean)
        self.assertEqual(expected_lens_spacy_src_mean, length_spacy.src.mean)
        self.assertEqual(expected_lens_spacy_tg_mean, length_spacy.tg.mean)

    def test_similarity(self):
        similarity = compute_similarity(self.dataset)
        expected_simi_mean = [1.0]
        expected_simi_pos = [1.0]
        self.assertEqual(expected_simi_mean, similarity.mean)
        self.assertEqual(expected_simi_pos, similarity.pos)


if __name__ == "__main__":
    unittest.main()
