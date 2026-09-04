import pickle
from pathlib import Path

import pandas as pd
from pycldf import Dataset

import cldf_repo
from cldf_repo import CLDFRepository
from clustering import neighbor_joining
from data_structures.guide_tree import TreeNode
from data_structures.profile import Profile
from lexstat_scores import get_lexstat_score
from msa import profile_alignment_algorithm
from msa.global_lexstat_model import GlobalLexstatModel
from simple_alignment import match_evaluator

model_path = Path("lexstat_model.pkl")
dist_path = Path("language_distances.pkl")

def build(ds: Dataset, concept: str) -> tuple[Profile, TreeNode]:
    cldf = cldf_repo.CLDFRepository(ds)

    if dist_path.exists() and model_path.exists():
        with open(model_path, "rb") as f:
            lexstat_model = pickle.load(f)
        lang_dist = pd.read_pickle(dist_path)
    else:
        lexstat_model = build_lexstat_matrices(cldf)
        with open(model_path, "wb") as f:
            pickle.dump(lexstat_model, f)

        lang_dist = build_language_distance_matrix(cldf, lexstat_model)
        lang_dist.to_pickle(dist_path)

    tree = neighbor_joining.build_tree_nj(lang_dist)

    profile = get_profile(tree, cldf, lexstat_model, concept)

    return profile, tree

def build_lexstat_matrices(cldf: CLDFRepository) -> GlobalLexstatModel:
    model = GlobalLexstatModel()

    language_list = sorted(cldf.get_all_language_names())

    for language in language_list:
        for language2 in language_list:
            if language == language2 or model.has_matrix(language, language2):
                continue

            print("Building Lexstat", language, "to", language2)

            lexstat_matrix = get_lexstat_score(cldf, language, language2)
            model.add_matrix(language, language2, lexstat_matrix)

    return model

def build_language_distance_matrix(cldf: CLDFRepository, lexstat_model: GlobalLexstatModel) -> pd.DataFrame:

    language_list = sorted(cldf.get_all_language_names())

    language_distances = pd.DataFrame(0.0, index=language_list, columns=language_list)

    for lang1 in language_list:
        for lang2 in language_list:
            if lang1 == lang2 or language_distances.at[lang1, lang2] != 0:
                continue

            print("Determining distance from", lang1, "to", lang2)

            lang1_id = cldf.find_language_id(lang1)
            lang2_id = cldf.find_language_id(lang2)

            word_tuples = cldf.get_same_meaning_pairs_as_tuples(lang1_id, lang2_id, 10000)

            accu_dist = 0.0
            count = 0

            for word_tuple in word_tuples:

                word1 = word_tuple[0].form
                word2 = word_tuple[1].form

                lexstat_matrix = lexstat_model.get_matrix(lang1, lang2)

                score, dist, _, _ = match_evaluator.evaluate_single(word1, word2, None, lexstat_matrix)
                accu_dist += dist
                count += 1

            if count > 0:
                avg_dist = accu_dist / count
            else:
                avg_dist = float("inf")

            language_distances.at[lang1, lang2] = avg_dist
            language_distances.at[lang2, lang1] = avg_dist

    return language_distances

def get_profile(tree_node: TreeNode, cldf: CLDFRepository, lexstat_model: GlobalLexstatModel, concept: str) -> Profile:

    if tree_node.is_leaf():
        lang_name = tree_node.name
        form = cldf.find_word(lang_name, concept).form
        return Profile.from_single_word(form, lang_name)

    left_profile = get_profile(tree_node.left, cldf, lexstat_model, concept)
    right_profile = get_profile(tree_node.right, cldf, lexstat_model, concept)

    profile, _, _, _, _ = profile_alignment_algorithm.align_profiles(left_profile, right_profile, False, False, None, lexstat_model)

    return profile