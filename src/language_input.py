import random
import warnings
from collections import defaultdict

from pycldf import Dataset
from pycldf.orm import Language

from src.environment.models import WordTuple


def find_language_id(dataset: Dataset, language_name: str) -> str | None:
    for lang in dataset.objects('LanguageTable'):
        if language_name.lower() == lang.cldf.name.lower() or language_name.lower() in lang.id.lower():
            return lang.id
    return None


def get_all_languages(dataset: Dataset) -> list[Language]:
    return list(dataset.objects('LanguageTable'))


def get_all_words_for_language(dataset: Dataset, language_name: str) -> list[str]:
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


def get_words_for_language_as_tuples(dataset: Dataset, language_name: str) -> list[WordTuple]:
    lang_id = find_language_id(dataset, language_name)
    if lang_id is None:
        return []

    target_lang_name = language_name
    for lang in dataset.objects('LanguageTable'):
        if lang.id == lang_id:
            target_lang_name = lang.cldf.name if lang.cldf.name else lang.id
            break

    # load all tuples from the entire dataset and filter them
    all_tuples = get_word_tuple_samples(dataset)
    return [word for word in all_tuples if word.language == target_lang_name]


def find_word_by_concept_string(dataset: Dataset, language_name: str, concept_string: str) -> str | None:

    lang_id = find_language_id(dataset, language_name)
    concept_id = find_concept_id(dataset, concept_string)

    if lang_id is None or concept_id is None:
        return None

    for form in dataset.objects('FormTable'):
        if form.cldf.languageReference == lang_id and form.cldf.parameterReference == concept_id:
            return extract_segments(form)
    return None


def find_concept_id(dataset: Dataset, concept_string: str) -> str | None:
    for param in dataset.objects('ParameterTable'):
        name = param.cldf.name if param.cldf.name else param.id
        if concept_string.lower() == name.lower():
            return param.id
    return None


def extract_segments(form) -> str:
    if form.cldf.segments:
        return "".join(form.cldf.segments).replace("+", "").replace("-", "")
    else:
        return form.cldf.form


def get_word_tuple_samples(dataset: Dataset, sample_ratio: float = 1.0, seed: int = 101) -> list[WordTuple]:

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


def get_all_concept_names(dataset: Dataset) -> list[str]:
    concept_names = []

    for param in dataset.objects('ParameterTable'):
        name = param.cldf.name if param.cldf.name else param.id
        if name and name not in concept_names:
            concept_names.append(name)

    return concept_names


def get_words_grouped_by_concept(dataset: Dataset) -> dict[str, list[WordTuple]]:
    lang_cache = {}
    for lang in dataset.objects('LanguageTable'):
        lang_cache[lang.id] = lang.cldf.name if lang.cldf.name else lang.id

    concept_cache = {}
    for param in dataset.objects('ParameterTable'):
        concept_cache[param.id] = param.cldf.name if param.cldf.name else param.id

    grouped_words = defaultdict(list)

    for form in dataset.objects('FormTable'):
        lang_name = lang_cache.get(form.cldf.languageReference, "Unknown_Language")
        concept_name = concept_cache.get(form.cldf.parameterReference, "Unknown_Concept")
        word_form = extract_segments(form)

        if word_form:
            word_tuple = WordTuple(
                language=lang_name,
                concept=concept_name,
                form=word_form
            )
            grouped_words[concept_name].append(word_tuple)

    return dict(grouped_words)


def get_noise_sample_for_language_pair(words_lang_a: list[WordTuple], words_lang_b: list[WordTuple],
                                       sample_size: int = 1000) -> list[tuple[WordTuple, WordTuple]]:

    noise_pairs = []
    attempts = 0

    max_attempts = sample_size * 20

    if not words_lang_a or not words_lang_b:
        return []

    while len(noise_pairs) < sample_size and attempts < max_attempts:
        word_a = random.choice(words_lang_a)
        word_b = random.choice(words_lang_b)

        if word_a.concept != word_b.concept:
            noise_pairs.append((word_a, word_b))

        attempts += 1

    return noise_pairs