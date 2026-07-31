import tomllib
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
config_path = root_dir / "environment/config.toml"

def _load_config() -> dict:
    with open(config_path, "rb") as f:
        return tomllib.load(f)

CONFIG = _load_config()