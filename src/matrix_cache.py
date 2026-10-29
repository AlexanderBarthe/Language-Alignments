import hashlib
import json
import math
import numbers
import os
import pickle
import tempfile
from pathlib import Path
from urllib.parse import urlparse


class PairMatrixCache:
    FORMAT_VERSION = 1
    COMPONENTS = ("FormTable", "LanguageTable", "ParameterTable")

    def __init__(self, cache_root: str | Path | None = None):
        self.cache_root = (
            Path(cache_root)
            if cache_root is not None
            else Path(__file__).resolve().parent.parent / "cache"
        )
        self.lexstat_dir = self.cache_root / "lexstat"
        self.distance_dir = self.cache_root / "language_distance"

    @classmethod
    def signature_for(cls, data, settings=None, calculation_version="1") -> str:
        dataset = getattr(data, "dataset", data)
        digest = hashlib.sha256()
        digest.update(f"pair-matrix-cache:{cls.FORMAT_VERSION}:{calculation_version}\0".encode())

        metadata = getattr(dataset, "metadata_dict", None)
        if metadata is None:
            metadata = getattr(dataset, "properties", {})
        cls._update_digest(digest, "metadata", cls._canonical_json(metadata).encode("utf-8"))

        components = getattr(dataset, "components", {})
        components = components if components is not None else {}
        for component in cls.COMPONENTS:
            table = components.get(component) if hasattr(components, "get") else None
            content = cls._component_content(dataset, component, table)
            cls._update_digest(digest, component, content)

        cls._update_digest(digest, "settings", cls._canonical_json(settings or {}).encode("utf-8"))
        return digest.hexdigest()

    @classmethod
    def signature_with_dependency(cls, base_signature: str, dependency) -> str:
        digest = hashlib.sha256()
        digest.update(f"pair-matrix-dependency:{cls.FORMAT_VERSION}\0".encode())
        cls._update_digest(digest, "base", base_signature.encode("utf-8"))
        cls._update_digest(digest, "dependency", cls._canonical_json(dependency).encode("utf-8"))
        return digest.hexdigest()

    def load_lexstat(self, lang_a: str, lang_b: str, signature: str) -> dict | None:
        payload = self._load_payload("lexstat", lang_a, lang_b, signature)
        if payload is None:
            return None

        matrix = self._valid_lexstat(payload["value"])
        if matrix is None:
            return None
        if lang_a <= lang_b:
            return matrix
        return {(sound_b, sound_a): score for (sound_a, sound_b), score in matrix.items()}

    def store_lexstat(self, lang_a: str, lang_b: str, matrix: dict, signature: str) -> None:
        valid_matrix = self._valid_lexstat(matrix)
        if valid_matrix is None:
            raise ValueError("Lexstat cache values must map sound pairs to numeric scores")
        if lang_a > lang_b:
            valid_matrix = {
                (sound_b, sound_a): score
                for (sound_a, sound_b), score in valid_matrix.items()
            }
        self._store_payload("lexstat", lang_a, lang_b, signature, valid_matrix)

    def load_distance(self, lang_a: str, lang_b: str, signature: str) -> float | None:
        payload = self._load_payload("distance", lang_a, lang_b, signature)
        if payload is None:
            return None
        distance = payload["value"]
        if not self._is_number(distance):
            return None
        value = float(distance)
        return None if math.isnan(value) else value

    def store_distance(self, lang_a: str, lang_b: str, distance: float, signature: str) -> None:
        if not self._is_number(distance) or math.isnan(float(distance)):
            raise ValueError("Distance cache values must be numeric and not NaN")
        self._store_payload("distance", lang_a, lang_b, signature, float(distance))

    def _load_payload(self, kind: str, lang_a: str, lang_b: str, signature: str) -> dict | None:
        if not isinstance(signature, str):
            return None
        path = self._entry_path(kind, lang_a, lang_b)
        try:
            with path.open("rb") as cache_file:
                payload = pickle.load(cache_file)
        except (
            OSError,
            pickle.PickleError,
            EOFError,
            AttributeError,
            ImportError,
            IndexError,
            TypeError,
            ValueError,
            RecursionError,
        ):
            return None

        if not isinstance(payload, dict) or set(payload) != {
            "format_version", "kind", "languages", "signature", "value"
        }:
            return None
        if payload["format_version"] != self.FORMAT_VERSION or payload["kind"] != kind:
            return None
        if payload["languages"] != self._canonical_pair(lang_a, lang_b):
            return None
        if payload["signature"] != signature:
            return None
        return payload

    def _store_payload(self, kind: str, lang_a: str, lang_b: str, signature: str, value) -> None:
        if not isinstance(signature, str):
            raise TypeError("Cache signatures must be strings")
        path = self._entry_path(kind, lang_a, lang_b)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": self.FORMAT_VERSION,
            "kind": kind,
            "languages": self._canonical_pair(lang_a, lang_b),
            "signature": signature,
            "value": value,
        }

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
            ) as cache_file:
                temp_path = Path(cache_file.name)
                pickle.dump(payload, cache_file, protocol=pickle.HIGHEST_PROTOCOL)
                cache_file.flush()
                os.fsync(cache_file.fileno())
            os.replace(temp_path, path)
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink()
                except FileNotFoundError:
                    pass

    def _entry_path(self, kind: str, lang_a: str, lang_b: str) -> Path:
        if kind == "lexstat":
            directory = self.lexstat_dir
        elif kind == "distance":
            directory = self.distance_dir
        else:
            raise ValueError(f"Unknown cache namespace: {kind}")
        pair_json = self._canonical_json(self._canonical_pair(lang_a, lang_b))
        identifier = hashlib.sha256(pair_json.encode("utf-8")).hexdigest()
        return directory / f"{identifier}.pkl"

    @staticmethod
    def _canonical_pair(lang_a: str, lang_b: str) -> tuple[str, str]:
        if not isinstance(lang_a, str) or not isinstance(lang_b, str):
            raise TypeError("Language labels must be strings")
        first, second = sorted((lang_a, lang_b))
        return first, second

    @classmethod
    def _valid_lexstat(cls, matrix) -> dict | None:
        if not isinstance(matrix, dict):
            return None
        result = {}
        for sound_pair, score in matrix.items():
            if (
                not isinstance(sound_pair, tuple)
                or len(sound_pair) != 2
                or not all(isinstance(sound, str) for sound in sound_pair)
                or not cls._is_number(score)
                or math.isnan(float(score))
            ):
                return None
            result[sound_pair] = float(score)
        return result

    @staticmethod
    def _is_number(value) -> bool:
        return isinstance(value, numbers.Real) and not isinstance(value, bool)

    @classmethod
    def _component_content(cls, dataset, component: str, table) -> bytes:
        if table is None:
            return b"<missing>"
        if table is not None:
            url = getattr(table, "url", "")
            url = getattr(url, "string", url)
            parsed_url = urlparse(str(url))
            directory = getattr(dataset, "directory", None)
            if isinstance(directory, (Path, str)) and not parsed_url.scheme:
                path = Path(url)
                if not path.is_absolute():
                    path = Path(directory) / path
                try:
                    return path.read_bytes()
                except OSError:
                    pass

        rows = list(dataset[component])
        return cls._canonical_json(rows).encode("utf-8")

    @staticmethod
    def _update_digest(digest, label: str, content: bytes) -> None:
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)

    @staticmethod
    def _canonical_json(value) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
