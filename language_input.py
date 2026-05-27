from pycldf import Dataset

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