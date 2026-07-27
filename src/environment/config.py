from pathlib import Path
import tomllib

root_dir = Path(__file__).resolve().parent.parent
config_path = root_dir / "config.toml"

def _load_config() -> dict:
    with open(config_path, "rb") as f:
        return tomllib.load(f)

CONFIG = _load_config()