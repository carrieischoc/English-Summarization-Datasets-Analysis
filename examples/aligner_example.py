"""
This example will extract the most relevant sentences from a reference text, based on a "summary text"
"""

from summaries.aligners import RougeNAligner


if __name__ == '__main__':

    aligner = RougeNAligner(n=2, optimization_attribute="fmeasure", lang="en")

    reference_text = ["This is a test document.",
                      "It contains multiple sentences.",
                      "Some of which are relevant for a summary.",
                      "The point of this program is to show which ones."]

    summary_text = ["This is another test document.",
                    "It contains fewer sentences."]

    print("The output of this script will print one sentence (as well as the similarity score and relative position) "
          "for each sentence in the summary.\n")
    for aligned_sentence in aligner.extract_source_sentences(summary_text, reference_text):
        print(aligned_sentence)
