import os
from pathlib import Path

import pycldf

import learning
import studies
from msa import language_tree, match_evaluator, profile_alignment_algorithm
from src import cldf_repo
from src.clustering import clustering
from src.lexstat_scores import get_lexstat_score

project_root = Path(__file__).resolve().parent.parent
lang_dir = os.path.join(project_root, "languages")

def main():

    ds = pycldf.Dataset.from_metadata("./languages/blumpanotacana/cldf/cldf-metadata.json")

    cldf = cldf_repo.CLDFRepository(ds)

    #studies.get_simple_alignment_success_rate(cldf)

    #print(studies.run_full_clust_learn_set(cldf))

    # load dataset and extract sequence samples
    #return

    # language_tree.print_lexstat_matrices(cldf)

    print(studies.get_simple_alignment_success_rate(cldf))

    return

    profile, tree = language_tree.build(ds, "sky", True, False)

    print(tree)
    print()
    print(profile)


def find_best_match(word_from_lang1, all_words_from_lang2):
    ds = pycldf.Dataset.from_metadata("./languages/blumpanotacana/cldf/cldf-metadata.json")
    cldf = cldf_repo.CLDFRepository(ds)

    lang1_name = "Shipibo-Konibo"
    lang2_name = "Tacana"
    concept = "Bridge"

    word = cldf.find_word(lang1_name, concept)
    word_from_lang1 = word.form if word is not None else None

    all_words_from_lang2 = cldf.get_all_words_for_language(lang2_name)

    if not word_from_lang1 or not all_words_from_lang2:
        return

    best_match, best_score, best_alignment, best_traceback, comparisons = match_evaluator.find_best_match(
        word_from_lang1, all_words_from_lang2, lang_a=lang1_name, lang_b=lang2_name)

    if best_match is None:
        return

    profile_alignment_algorithm.print_matrix(best_alignment, word_from_lang1, best_match)
    print()
    profile_alignment_algorithm.print_matrix(best_traceback, word_from_lang1, best_match)
    print()
    print(profile_alignment_algorithm.align(
        word_from_lang1, lang1_name, best_match, lang2_name, False, False)[0])
    print()

    print("Original word: ", word_from_lang1)
    print("Best match: ", best_match)
    print("Best Score: ", best_score)
    print("Comparisons: ", comparisons)

def cluster():
    ds = pycldf.Dataset.from_metadata("./languages/blumpanotacana/cldf/cldf-metadata.json")

    cldf = cldf_repo.CLDFRepository(ds)
    word_list = cldf.get_word_tuples(sample_ratio=0.4, seed=101)

    df = match_evaluator.match_every_to_distance(word_list)

    cluster_frame, tree = clustering.run_hierarchical_clustering(df)

    impurity = clustering.calculate_cluster_impurity(cluster_frame)

    print("Impurity: ", impurity)
    print("Cluster amount: ", clustering.get_cluster_count(cluster_frame))

    for i in range(clustering.get_cluster_count(cluster_frame)):
        print("Entries of cluster", i)
        print(clustering.get_entries_from_cluster(cluster_frame, i))

def optimize_align_params():
    learning.find_best_alignment_params()

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