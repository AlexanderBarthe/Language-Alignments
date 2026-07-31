import random
from collections import defaultdict

from pycldf import Dataset
from pycldf.orm import Language

from src.data_structures.models import WordTuple


class CLDFRepository:
    def __init__(self, dataset: Dataset, preload_caches: bool = True):
        self.dataset = dataset

        # internal cache dictionaries for o(1) lookups
        self._lang_id_to_name: dict[str, str] = {}
        self._concept_id_to_name: dict[str, str] = {}
        self._lang_name_to_id: dict[str, str] = {}
        self._concept_name_to_id: dict[str, str] = {}

        if preload_caches:
            self.refresh_caches()

    def refresh_caches(self) -> None:
        # build language lookups
        self._lang_id_to_name.clear()
        self._lang_name_to_id.clear()

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
    def _extract_segments(form) -> str:
        # clean and join segments or return raw form
        if form.cldf.segments:
            return "".join(form.cldf.segments).replace("+", "").replace("-", "")
        else:
            return form.cldf.form

    def get_all_languages(self) -> list[Language]:
        return list(self.dataset.objects('LanguageTable'))

    def get_all_language_names(self) -> list[str]:
        return list(self._lang_id_to_name.values())

    def find_language_id(self, language_name: str) -> str | None:
        return self._lang_name_to_id.get(language_name.lower())

    def get_all_words_for_language(self, language_name: str) -> list[str]:
        tuples = self.get_words_for_language_as_tuples(language_name)
        return [word.form for word in tuples]

    def get_words_for_language_as_tuples(self, language_name: str) -> list[WordTuple]:
        return self.get_word_tuples(languages=[language_name])

    def get_all_concept_names(self) -> list[str]:
        return list(self._concept_id_to_name.values())

    def find_concept_id(self, concept_string: str) -> str | None:
        return self._concept_name_to_id.get(concept_string.lower())

    def get_all_words_for_concept(self, concept_string: str) -> list[str]:
        tuples = self.get_words_for_concept_as_tuples(concept_string)
        return [word.form for word in tuples]

    def get_words_for_concept_as_tuples(self, concept_string: str) -> list[WordTuple]:
        return self.get_word_tuples(concepts=[concept_string])

    def get_words_grouped_by_concept(self) -> dict[str, list[WordTuple]]:
        grouped_words = defaultdict(list)
        for word_tuple in self.get_word_tuples():
            grouped_words[word_tuple.concept].append(word_tuple)
        return dict(grouped_words)

    def find_word(self, language_name: str, concept_string: str) -> WordTuple | None:
        results = self.get_word_tuples(languages=[language_name], concepts=[concept_string])
        if results:
            return results[0]
        return None

    def get_word_tuples(self, languages: list[str] = None, concepts: list[str] = None,
                        sample_ratio: float = 1.0, seed: int = 101) -> list[WordTuple]:
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

        # apply sampling if no strict language filter was provided
        if sample_ratio < 1.0 and allowed_lang_ids is None:
            all_lang_ids = list(self._lang_id_to_name.keys())
            random.seed(seed)
            num_to_keep = max(1, int(len(all_lang_ids) * sample_ratio))
            allowed_lang_ids = set(random.sample(all_lang_ids, num_to_keep))

        results = []

        # single pass through the form table
        for form in self.dataset.objects('FormTable'):
            lang_id = form.cldf.languageReference
            concept_id = form.cldf.parameterReference

            if allowed_lang_ids is not None and lang_id not in allowed_lang_ids:
                continue

            if allowed_concept_ids is not None and concept_id not in allowed_concept_ids:
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

    def get_same_meaning_pairs_as_tuples(self, lang1_id: str, lang2_id: str, max_size: int = 200) -> list[tuple[WordTuple, WordTuple]]:
        # resolve language names from ids
        lang1_name = self._lang_id_to_name.get(lang1_id)
        lang2_name = self._lang_id_to_name.get(lang2_id)
        if not lang1_name or not lang2_name:
            return []

        # fetch word tuples for both languages
        words1 = self.get_word_tuples(languages=[lang1_name])
        words2 = self.get_word_tuples(languages=[lang2_name])

        # index words by concept
        map1 = {w.concept: w for w in words1}
        map2 = {w.concept: w for w in words2}

        # find shared concepts and build pairs
        common_concepts = set(map1.keys()) & set(map2.keys())
        pairs = [(map1[c], map2[c]) for c in common_concepts]

        # sample or return n pairs
        if len(pairs) > max_size:
            return random.sample(pairs, max_size)
        return pairs

    @staticmethod
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


