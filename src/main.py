import pycldf

import alignment_algorithm
import clustering
import language_input
import match_evaluator
import parameter_optimization


def main():

    ds = pycldf.Dataset.from_metadata("./languages/blumpanotacana/cldf/cldf-metadata.json")

    word_list = language_input.get_all_words_as_tuples(ds, 0.1, 101)

    match_evaluator.show_top_matches(word_list, 1000)

    #parameter_optimization.find_best_dbscan_params()

def find_best_match(word_from_lang1, all_words_from_lang2):
    ds = pycldf.Dataset.from_metadata("./languages/blumpanotacana/cldf/cldf-metadata.json")

    lang1_name = "Shipibo"
    lang2_name = "Tacana"
    concept = "Bridge"

    word_from_lang1 = language_input.find_word_by_concept_string(ds, lang1_name, concept)

    all_words_from_lang2 = language_input.get_all_words(ds, lang2_name)

    if not word_from_lang1 or not all_words_from_lang2:
        return

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

def cluster():
    ds = pycldf.Dataset.from_metadata("./languages/blumpanotacana/cldf/cldf-metadata.json")

    word_list = language_input.get_all_words_as_tuples(ds, 0.4, 101)

    df = match_evaluator.match_every(word_list)

    cluster_frame, tree = clustering.run_hierarchical_clustering(df)

    impurity = clustering.calculate_cluster_impurity(cluster_frame)

    print("Impurity: ", impurity)
    print("Cluster amount: ", clustering.get_cluster_count(cluster_frame))
    print("Entries of cluster 1: ", clustering.get_entries_from_cluster(cluster_frame, 1))

def optimize_align_params():
    parameter_optimization.find_best_alignment_params()

if __name__ == "__main__":
    main()