from itertools import combinations

import pandas as pd
from pycldf import Dataset

from cldf_repo import CLDFRepository
from language_input import get_word_tuple_samples
from learning import run_optimization_suite
from simple_alignment import match_evaluator


def get_simple_alignment_success_rate(cldf_repo: CLDFRepository) -> None | float:
    lang_names = cldf_repo.get_all_language_names()

    word_matches = 0
    word_accu = 0

    for lang1_name, lang2_name in combinations(lang_names, 2):
        lang1_id = cldf_repo.find_language_id(lang1_name)
        lang2_id = cldf_repo.find_language_id(lang2_name)

        if not lang1_id or not lang2_id:
            continue

        print(f"Matching {lang1_name} to {lang2_name}")

        word_tuples = cldf_repo.get_same_meaning_pairs_as_tuples(lang1_id, lang2_id)
        all_words_lang2 = cldf_repo.get_all_words_for_language(lang2_name)

        for word_tuple in word_tuples:
            word1 = word_tuple[0]
            word2 = word_tuple[1]

            best_match, _, _, _, _ = match_evaluator.find_best_match(word1.form, all_words_lang2)

            print(f"Original word: {word1.form}, Best match: {best_match}, Actual word: {word2.form}")

            if best_match == word2.form:
                word_matches += 1

            word_accu += 1

        print(f"Matches: {word_matches}")
        print(f"Accu: {word_accu}")


    if word_accu == 0:
        return None

    return word_matches / word_accu


def run_full_clust_learn_set (
        ds: Dataset,
        output_csv_path: str = "clustering_evaluation_results.csv",
        n_trials: int = 50,
        sample_ratio: float = 0.2,
        seed: int = 100
) -> pd.DataFrame:
    sequences_sample = get_word_tuple_samples(ds, sample_ratio=sample_ratio, seed=seed)

    algos_to_test = [
        'hdbscan',
        'dynamic_tree',
        'infomap',
        'kmedoids',
        'affinity_propagation'
    ]
    modes_to_test = ['alignment', 'clustering', 'joint']

    print(f"starting evaluation on {len(sequences_sample)} sequences.")
    print(f"algorithms: {algos_to_test}")
    print(f"modes: {modes_to_test}")
    print(f"trials per combination: {n_trials}")
    print("-" * 50)

    # execute all combinations using the optimization suite
    results_df = run_optimization_suite(
        sequences=sequences_sample,
        algorithms=algos_to_test,
        modes=modes_to_test,
        n_trials=n_trials
    )

    results_df = results_df.sort_values(by=['algorithm', 'mode'])
    results_df.to_csv(output_csv_path, index=False)

    print("-" * 50)
    print(f"evaluation finished. results saved to: {output_csv_path}\n")

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(results_df.to_string(index=False))

    return results_df