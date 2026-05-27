import alignment_algorithm
import language_input
import match_evaluator
import pycldf

def main():

    ds = pycldf.Dataset.from_metadata("languages/blumpanotacana/cldf/cldf-metadata.json")

    lang1_name = "Shipibo"
    lang2_name = "Tacana"
    concept = "Bridge"

    word_from_lang1 = language_input.find_word_by_concept_string(ds, lang1_name, concept)

    all_words_from_lang2 = language_input.get_all_words(ds, lang2_name)

    if word_from_lang1 and all_words_from_lang2:
        find_best_match(word_from_lang1, all_words_from_lang2)


def find_best_match(word_from_lang1, all_words_from_lang2):
    best_match, best_score, best_alignment, best_traceback, comparisons = match_evaluator.find_best_match(
        word_from_lang1, all_words_from_lang2)

    alignment_algorithm.print_matrix(best_alignment, "-" + word_from_lang1, "-" + best_match)
    print()
    alignment_algorithm.print_matrix(best_traceback, "-" + word_from_lang1, "-" + best_match)
    print()
    alignment_algorithm.print_alignment(best_traceback, "-" + word_from_lang1, "-" + best_match)
    print()

    print("Original word: ", word_from_lang1)
    print("Best match: ", best_match)
    print("Best Score: ", best_score)
    print("Comparisons: ", comparisons)

if __name__ == "__main__":
    main()