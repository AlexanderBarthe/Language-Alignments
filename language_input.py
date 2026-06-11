import random
from typing import NamedTuple

from pycldf import Dataset


class WordTuple(NamedTuple):
    language: str
    concept: str
    form: str

def find_language_id(dataset: Dataset, language_name: str):
    for lang in dataset.objects('LanguageTable'):
        if language_name.lower() == lang.cldf.name.lower() or language_name.lower() in lang.id.lower():
            return lang.id
    return None

def get_all_words(dataset: Dataset, language_name: str):
    all_words = []

    lang_id = find_language_id(dataset, language_name)
    if lang_id is None:
        return []

    for form in dataset.objects('FormTable'):
        if form.cldf.languageReference == lang_id:
            word = extract_segments(form)
            if word:
                all_words.append(word)

    return all_words

def find_word_by_concept_string(dataset: Dataset, language_name: str, concept_string: str):

    lang_id = find_language_id(dataset, language_name)
    concept_id = find_concept_id(dataset, concept_string)

    if lang_id is None or concept_id is None:
        return None

    for form in dataset.objects('FormTable'):
        if form.cldf.languageReference == lang_id and form.cldf.parameterReference == concept_id:
            return extract_segments(form)
    return None

def find_concept_id(dataset: Dataset, concept_string: str):
    for param in dataset.objects('ParameterTable'):
        name = param.cldf.name if param.cldf.name else param.id
        if concept_string.lower() == name.lower():
            return param.id
    return None


def extract_segments(form):
    if form.cldf.segments:
        return "".join(form.cldf.segments).replace("+", "").replace("-", "")
    else:
        return form.cldf.form


def get_all_words_as_tuples(dataset: Dataset, sample_ratio: float = 1.0, seed: int = 101) -> list[WordTuple]:

    all_languages = list(dataset.objects('LanguageTable'))

    # Select subset of all languages
    if sample_ratio < 1.0:
        random.seed(seed)

        num_to_keep = int(len(all_languages) * sample_ratio)

        num_to_keep = max(1, num_to_keep)

        selected_languages = random.sample(all_languages, num_to_keep)
    else:
        selected_languages = all_languages

    lang_cache = {}
    for lang in selected_languages:
        lang_cache[lang.id] = lang.cldf.name if lang.cldf.name else lang.id

    concept_cache = {}
    for param in dataset.objects('ParameterTable'):
        concept_cache[param.id] = param.cldf.name if param.cldf.name else param.id

    all_word_tuples = []

    for form in dataset.objects('FormTable'):
        if form.cldf.languageReference not in lang_cache:
            continue

        lang_name = lang_cache[form.cldf.languageReference]
        concept_name = concept_cache.get(form.cldf.parameterReference, "Unknown_Concept")

        word_form = extract_segments(form)

        if word_form:
            all_word_tuples.append(WordTuple(
                language=lang_name,
                concept=concept_name,
                form=word_form
            ))

    return all_word_tuples