import math
import random
from collections import defaultdict

from pycldf import Dataset
from pycldf.orm import Language

from config import CONFIG
from data_structures.phonemes import tokenize
from src.data_structures.models import (
    ConceptId,
    ConceptName,
    LangId,
    LangName,
    WordForm,
    WordTuple,
)


class CLDFRepository:
    def __init__(self, dataset: Dataset, preload_caches: bool = True):
        self.dataset = dataset

        # internal cache dictionaries for o(1) lookups
        self._lang_id_to_name: dict[LangId, LangName] = {}
        self._concept_id_to_name: dict[ConceptId, ConceptName] = {}
        self._lang_name_to_id: dict[str, LangId] = {}
        self._concept_name_to_id: dict[str, ConceptId] = {}
        self._usable_concept_ids: set[ConceptId] = set()

        self.min_concept_coverage = CONFIG['cldf']['min_concept_coverage']
        if not 0.0 <= self.min_concept_coverage <= 1.0:
            self.min_concept_coverage = 0.9

        if preload_caches:
            self.refresh_caches()

    def refresh_caches(self) -> None:
        # build language lookups
        self._lang_id_to_name.clear()
        self._lang_name_to_id.clear()
        self._usable_concept_ids = set()

        for lang in self.dataset.objects('LanguageTable'):
            name = lang.cldf.name if lang.cldf.name else lang.id
            self._lang_id_to_name[lang.id] = name
            self._lang_name_to_id[name.lower()] = lang.id
            self._lang_name_to_id[lang.id.lower()] = lang.id

        # build concept lookups
        self._concept_id_to_name.clear()
        self._concept_name_to_id.clear()

        for param in self.dataset.objects('ParameterTable'):
            name = param.cldf.name if param.cldf.name else param.id
            self._concept_id_to_name[param.id] = name
            self._concept_name_to_id[name.lower()] = param.id

    @staticmethod
    def _extract_segments(form) -> WordForm:
        # preserve existing segment boundaries or tokenize the raw form
        if form.cldf.segments:
            return tuple(token for token in tokenize(form.cldf.segments) if token != "-")
        else:
            return tuple(token for token in tokenize(form.cldf.form or "") if token != "-")

    def get_all_languages(self) -> list[Language]:
        return list(self.dataset.objects('LanguageTable'))

    def get_all_language_names(self) -> list[LangName]:
        return list(self._lang_id_to_name.values())

    def find_language_id(self, language_name: LangName) -> LangId | None:
        return self._lang_name_to_id.get(language_name.lower())

    def get_all_words_for_language(self, language_name: LangName, usable_only: bool = True) -> list[WordForm]:
        tuples = self.get_words_for_language_as_tuples(
            language_name,
            usable_only=usable_only
        )
        return [word.form for word in tuples]

    def get_words_for_language_as_tuples(self, language_name: LangName, usable_only: bool = True) -> list[WordTuple]:
        return self.get_word_tuples(
            languages=[language_name],
            usable_only=usable_only
        )

    def get_all_concept_names(self) -> list[ConceptName]:
        return list(self._concept_id_to_name.values())

    def find_concept_id(self, concept_string: ConceptName) -> ConceptId | None:
        return self._concept_name_to_id.get(concept_string.lower())

    def get_all_words_for_concept(self, concept_string: ConceptName, usable_only: bool = True) -> list[WordForm]:
        tuples = self.get_words_for_concept_as_tuples(
            concept_string,
            usable_only=usable_only
        )
        return [word.form for word in tuples]

    def get_words_for_concept_as_tuples(self, concept_string: ConceptName, usable_only: bool = True) -> list[WordTuple]:
        return self.get_word_tuples(
            concepts=[concept_string],
            usable_only=usable_only
        )

    def get_words_grouped_by_concept(self, usable_only: bool = True) -> dict[ConceptName, list[WordTuple]]:
        grouped_words = defaultdict(list)

        for word_tuple in self.get_word_tuples(usable_only=usable_only):
            grouped_words[word_tuple.concept].append(word_tuple)

        return dict(grouped_words)

    def find_word(self, language_name: LangName, concept_string: ConceptName, usable_only: bool = True ) -> WordTuple | None:
        results = self.get_word_tuples(
            languages=[language_name],
            concepts=[concept_string],
            usable_only=usable_only
        )

        if results:
            return results[0]

        return None

    def _get_usable_concept_ids(self) -> set[ConceptId]:
        if len(self._usable_concept_ids) > 0:
            return self._usable_concept_ids

        total_languages = len(self._lang_id_to_name)

        if total_languages == 0:
            self._usable_concept_ids = set()
            return self._usable_concept_ids

        languages_per_concept: dict[ConceptId, set[LangId]] = defaultdict(set)

        for form in self.dataset.objects('FormTable'):
            if not self._extract_segments(form):
                continue

            languages_per_concept[form.cldf.parameterReference].add(
                form.cldf.languageReference
            )

        min_language_count = math.ceil(
            total_languages * self.min_concept_coverage
        )

        self._usable_concept_ids = {
            concept_id
            for concept_id, concept_languages in languages_per_concept.items()
            if len(concept_languages) >= min_language_count
        }

        return self._usable_concept_ids

    def get_word_tuples(self, languages: list[LangName] | None = None, concepts: list[ConceptName] | None = None,
                        sample_ratio: float = 1.0, seed: int = 101, usable_only: bool = True) -> list[WordTuple]:
        # resolve language filters
        allowed_lang_ids = None
        if languages is not None:
            allowed_lang_ids = {self.find_language_id(lang) for lang in languages}
            allowed_lang_ids.discard(None)

        # resolve concept filters
        allowed_concept_ids = None
        if concepts is not None:
            allowed_concept_ids = {self.find_concept_id(concept) for concept in concepts}
            allowed_concept_ids.discard(None)

        # resolve usable concepts
        usable_concept_ids = (
            self._get_usable_concept_ids()
            if usable_only
            else None
        )

        # apply sampling if no strict language filter was provided
        if sample_ratio < 1.0 and allowed_lang_ids is None:
            all_lang_ids = list(self._lang_id_to_name.keys())
            rng = random.Random(seed)
            num_to_keep = max(1, int(len(all_lang_ids) * sample_ratio))
            allowed_lang_ids = set(rng.sample(all_lang_ids, num_to_keep))

        results = []

        # single pass through the form table
        for form in self.dataset.objects('FormTable'):
            lang_id = form.cldf.languageReference
            concept_id = form.cldf.parameterReference

            if allowed_lang_ids is not None and lang_id not in allowed_lang_ids:
                continue

            if allowed_concept_ids is not None and concept_id not in allowed_concept_ids:
                continue

            if usable_concept_ids is not None and concept_id not in usable_concept_ids:
                continue

            word_form = self._extract_segments(form)
            if word_form:
                lang_name = self._lang_id_to_name.get(lang_id, "Unknown_Language")
                concept_name = self._concept_id_to_name.get(concept_id, "Unknown_Concept")

                results.append(WordTuple(
                    language=lang_name,
                    concept=concept_name,
                    form=word_form
                ))

        return results

    def get_usable_concepts(self) -> list[ConceptName]:
        return sorted(
            self._concept_id_to_name.get(concept_id, "Unknown_Concept")
            for concept_id in self._get_usable_concept_ids()
        )

    def get_same_meaning_pairs_as_tuples(
            self,
            lang1_id: LangId,
            lang2_id: LangId,
            sampled_concepts: list[str] | None = None,
            max_size: int = 10000,
            usable_only: bool = True,
            random_state: int | str | None = None
    ) -> list[tuple[WordTuple, WordTuple]]:

        lang1_name = self._lang_id_to_name.get(lang1_id)
        lang2_name = self._lang_id_to_name.get(lang2_id)

        if not lang1_name or not lang2_name:
            return []

        words1 = self.get_word_tuples(
            languages=[lang1_name],
            usable_only=usable_only
        )

        words2 = self.get_word_tuples(
            languages=[lang2_name],
            usable_only=usable_only
        )

        map1 = {w.concept: w for w in words1}
        map2 = {w.concept: w for w in words2}

        if sampled_concepts is not None:
            return [
                (map1[concept], map2[concept])
                for concept in sampled_concepts
                if concept in map1 and concept in map2
            ]

        common_concepts = sorted(set(map1.keys()) & set(map2.keys()))

        pairs = [(map1[c], map2[c]) for c in common_concepts]

        if random_state is None:
            random_state = f"{lang1_id}::{lang2_id}"

        rng = random.Random(random_state)

        if len(pairs) > max_size:
            return rng.sample(pairs, max_size)

        return pairs

    @staticmethod
    def get_noise_sample_for_language_pair(words_lang_a: list[WordTuple], words_lang_b: list[WordTuple],
                                           sample_size: int = 1000,
                                           random_state: int | str | None = None) -> list[tuple[WordTuple, WordTuple]]:
        noise_pairs = []
        attempts = 0
        max_attempts = sample_size * 20

        if not words_lang_a or not words_lang_b:
            return []

        if random_state is None:
            random_state = f"{words_lang_a[0].language}::{words_lang_b[0].language}"

        rng = random.Random(random_state)

        while len(noise_pairs) < sample_size and attempts < max_attempts:
            word_a = rng.choice(words_lang_a)
            word_b = rng.choice(words_lang_b)

            if word_a.concept != word_b.concept:
                noise_pairs.append((word_a, word_b))

            attempts += 1

        return noise_pairs