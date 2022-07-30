import unittest

from inspection import get_print_lens
from similarity import get_print_simi


class TestStats(unittest.TestCase):

    def test_length(self):
        source = []

    def test_similarity(self):
        aligner = RougeNAligner(n=2, optimization_attribute="fmeasure")

        # First test by passing sentencized inputs
        gold = ["This is a test.", "This is another, worse test."]
        system = ["This is a test."]

        result = aligner.extract_source_sentences(system, gold)
        expected_result = [RelevantSentence(gold[0], 1.0, 0.0)]
        self.assertEqual(expected_result, result)

if __name__ == '__main__':
    unittest.main()