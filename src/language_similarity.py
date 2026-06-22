import pycldf
from pycldf import Dataset

import language_input
import pandas as pd
import numpy as np

from src import match_evaluator
from src.clustering import run_hierarchical_clustering


def score_words_conceptwise(ds: Dataset) -> dict[str, pd.DataFrame]:
    words = language_input.get_words_grouped_by_concept(ds)

    output: dict[str, pd.DataFrame] = {}

    for concept in words.keys():

        scoring = match_evaluator.match_every(words[concept])

        output[concept] = scoring

    return output


def get_language_closeness_matrix(ds: Dataset) -> pd.DataFrame:
    concept_distances = score_words_conceptwise(ds)

    # Extract languages
    scored_languages = set()
    for df_matrix in concept_distances.values():
        for tuple_i in df_matrix.index:
            scored_languages.add(tuple_i[0])

    all_languages = sorted(list(scored_languages))
    num_langs = len(all_languages)


    assoc_matrix = pd.DataFrame(
        np.zeros((num_langs, num_langs)),
        index=all_languages,
        columns=all_languages
    )

    concept_count = 0

    for concept, df_matrix in concept_distances.items():

        results_df, tree = run_hierarchical_clustering(df_matrix, cutoff=0.4, method='average')

        # Build mapping word Tuple to Cluster_ID
        cluster_mapping = dict(zip(df_matrix.index, results_df['Cluster_ID']))

        for tuple_i in df_matrix.index:
            for tuple_j in df_matrix.columns:

                # Increase language score if in same map
                if cluster_mapping[tuple_i] == cluster_mapping[tuple_j]:
                    lang_i = tuple_i[0]
                    lang_j = tuple_j[0]

                    assoc_matrix.loc[lang_i, lang_j] += 1

        concept_count += 1

    return assoc_matrix / concept_count


def print_top_similarities(assoc_matrix: pd.DataFrame, n = 3):

    for lang in assoc_matrix.index:

        row = assoc_matrix.loc[lang]

        # Delete relation to current language
        filtered_row = row.drop(labels=[lang])

        top_n = filtered_row.sort_values(ascending=False).head(n)

        print(f"Language: {lang}")
        for other_lang, score in top_n.items():
            print(f" - {other_lang}: {score * 100:.1f}% shared clusters")
        print("-" * 40)