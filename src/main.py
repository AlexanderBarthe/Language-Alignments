import os
from pathlib import Path

import pycldf

import language_input
import parameter_optimization
from msa import language_tree
from src import cldf_repo
from src.clustering import clustering
from src.lexstat_scores import get_lexstat_score
from src.simple_alignment import alignment_algorithm, match_evaluator

project_root = Path(__file__).resolve().parent.parent
lang_dir = os.path.join(project_root, "languages")

def main():

    ds = pycldf.Dataset.from_metadata("./languages/blumpanotacana/cldf/cldf-metadata.json")

    profile, tree = language_tree.build(ds, "sky")

    print(tree)
    print()
    print(profile)


def find_best_match(word_from_lang1, all_words_from_lang2):
    ds = pycldf.Dataset.from_metadata("./languages/blumpanotacana/cldf/cldf-metadata.json")

    lang1_name = "Shipibo"
    lang2_name = "Tacana"
    concept = "Bridge"

    word_from_lang1 = language_input.find_word(lang1_name, concept)

    all_words_from_lang2 = language_input.get_all_words_for_language(ds, lang2_name)

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

    word_list = language_input.get_word_tuple_samples(ds, 0.4, 101)

    df = match_evaluator.match_every_to_distance(word_list)

    cluster_frame, tree = clustering.run_hierarchical_clustering(df)

    impurity = clustering.calculate_cluster_impurity(cluster_frame)

    print("Impurity: ", impurity)
    print("Cluster amount: ", clustering.get_cluster_count(cluster_frame))

    for i in range(clustering.get_cluster_count(cluster_frame)):
        print("Entries of cluster", i)
        print(clustering.get_entries_from_cluster(cluster_frame, i))

def optimize_align_params():
    parameter_optimization.find_best_alignment_params()

def cluster_two_langs_with_lexstat_scoring():
    ds = pycldf.Dataset.from_metadata("./languages/blumpanotacana/cldf/cldf-metadata.json")

    cldf = cldf_repo.CLDFRepository(ds)

    lang1_name = "Shipibo-Konibo"
    lang2_name = "Tacana"

    words = cldf.get_words_for_language_as_tuples(lang1_name)
    words.extend(cldf.get_words_for_language_as_tuples(lang2_name))

    lexstat_scores = get_lexstat_score(ds, lang1_name, lang2_name)

    df = match_evaluator.match_every_to_distance(words, None, lexstat_scores)



    cluster_frame = clustering.run_dbscan_clustering(df, 0.1)

    impurity = clustering.calculate_cluster_impurity(cluster_frame)

    print("Impurity: ", impurity)
    print("Cluster amount: ", clustering.get_cluster_count(cluster_frame))
    for i in range(clustering.get_cluster_count(cluster_frame)):
        print(clustering.get_entries_from_cluster(cluster_frame, i))


if __name__ == "__main__":
    main()