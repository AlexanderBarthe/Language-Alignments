from itertools import combinations

from cldf_repo import CLDFRepository
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
