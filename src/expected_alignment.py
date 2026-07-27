from pycldf import Dataset

from src import language_input


def get_random_vocal_matching(ds: Dataset, language_name_a: str, language_name_b: str):

    words_lang_a = language_input.get_words_for_language_as_tuples(ds, language_name_a)
    words_lang_b = language_input.get_words_for_language_as_tuples(ds, language_name_b)

    noise_samples = language_input.get_noise_sample_for_language_pair(words_lang_a, words_lang_b)

    