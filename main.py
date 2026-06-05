import alignment_algorithm
import match_evaluator

def main():

    seq1 = 'XQWERTZX'
    seq2 = 'XRTZQWEX'

    score, fs_i, fs_j, matrix, traceback = match_evaluator.evaluate_single(seq1, seq2)

    alignment_algorithm.print_matrix(matrix, "-" + seq1, "-" + seq2)
    print()
    alignment_algorithm.print_matrix(traceback, "-" + seq1, "-" + seq2)
    print()
    alignment_algorithm.print_alignment(traceback, "-" + seq1, "-" + seq2)
    print()

    print(score)

    '''
    ds = pycldf.Dataset.from_metadata("languages/blumpanotacana/cldf/cldf-metadata.json")

    lang1_name = "Shipibo"
    lang2_name = "Tacana"
    concept = "Bridge"

    word_from_lang1 = language_input.find_word_by_concept_string(ds, lang1_name, concept)

    all_words_from_lang2 = language_input.get_all_words(ds, lang2_name)

    if word_from_lang1 and all_words_from_lang2:
        find_best_match(word_from_lang1, all_words_from_lang2)
    '''

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