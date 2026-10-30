import random
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import combinations

import pandas as pd
from tqdm import tqdm

from cldf_repo import CLDFRepository
from data_structures.models import WordForm, WordTuple
from config import CONFIG
from learning import run_optimization_suite
from msa import match_evaluator


def _prepare_language_pair_work(
        cldf_repo: CLDFRepository,
) -> tuple[int, Iterator[tuple[list[tuple[WordTuple, WordTuple]], list[WordForm]]]]:
    words_by_language: dict[str, list[WordTuple]] = defaultdict(list)
    for word in cldf_repo.get_word_tuples():
        words_by_language[word.language].append(word)

    language_names = cldf_repo.get_all_language_names()
    words_by_concept = {
        language: {word.concept: word for word in words_by_language[language]}
        for language in language_names
    }
    language_ids = {name: cldf_repo.find_language_id(name) for name in language_names}
    eligible_language_pairs: list[tuple[str, str]] = []
    for lang1_name, lang2_name in combinations(language_names, 2):
        if not language_ids[lang1_name] or not language_ids[lang2_name]:
            continue

        shared_concepts = words_by_concept[lang1_name].keys() & words_by_concept[lang2_name].keys()
        if shared_concepts:
            eligible_language_pairs.append((lang1_name, lang2_name))

    return len(eligible_language_pairs), _iter_language_pair_work(
        words_by_language, words_by_concept, eligible_language_pairs)


def _iter_language_pair_work(
        words_by_language: dict[str, list[WordTuple]],
        words_by_concept: dict[str, dict[str, WordTuple]],
        eligible_language_pairs: list[tuple[str, str]],
) -> Iterator[tuple[list[tuple[WordTuple, WordTuple]], list[WordForm]]]:
    for lang1_name, lang2_name in eligible_language_pairs:
        words1_by_concept = words_by_concept[lang1_name]
        words2_by_concept = words_by_concept[lang2_name]
        shared_concepts = words1_by_concept.keys() & words2_by_concept.keys()
        word_pairs = [
            (words1_by_concept[concept], words2_by_concept[concept])
            for concept in shared_concepts
        ]

        if len(word_pairs) > 200:
            word_pairs = random.sample(word_pairs, 200)
        candidate_forms = [word.form for word in words_by_language[lang2_name]]
        yield word_pairs, candidate_forms


def _score_language_pair(word_pairs: list[tuple[WordTuple, WordTuple]],
                         candidate_forms: list[WordForm]) -> tuple[int, int]:
    matches = 0
    comparisons = 0

    for word1, word2 in word_pairs:
        best_match, _, _, _, _ = match_evaluator.find_best_match(
            word1.form, candidate_forms, lang_a=word1.language, lang_b=word2.language)
        if best_match == word2.form:
            matches += 1
        comparisons += 1

    return matches, comparisons


def get_simple_alignment_success_rate(cldf_repo: CLDFRepository) -> None | float:
    word_matches = 0
    comparisons = 0
    max_workers = CONFIG["alignment"]["mt_workers"]
    eligible_pair_count, work_items = _prepare_language_pair_work(cldf_repo)
    if eligible_pair_count == 0:
        return None

    with tqdm(total=eligible_pair_count, desc="Matching language pairs") as progress_bar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            pending_futures = set()

            for _ in range(max_workers):
                try:
                    word_pairs, candidate_forms = next(work_items)
                except StopIteration:
                    break
                pending_futures.add(executor.submit(_score_language_pair, word_pairs, candidate_forms))

            while pending_futures:
                completed_futures, pending_futures = wait(
                    pending_futures, return_when=FIRST_COMPLETED)
                for future in completed_futures:
                    pair_matches, pair_comparisons = future.result()
                    word_matches += pair_matches
                    comparisons += pair_comparisons
                    progress_bar.update(1)

                    try:
                        word_pairs, candidate_forms = next(work_items)
                    except StopIteration:
                        continue
                    pending_futures.add(executor.submit(_score_language_pair, word_pairs, candidate_forms))


    if comparisons == 0:
        return None

    return word_matches / comparisons


def run_full_clust_learn_set (
        cldf: CLDFRepository,
        output_csv_path: str = "clustering_evaluation_results.csv",
        n_trials: int = 50,
        sample_ratio: float = 0.2,
        seed: int = 100
) -> pd.DataFrame:
    sequences_sample = cldf.get_word_tuples(sample_ratio=sample_ratio, seed=seed)

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