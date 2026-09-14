from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    api_version: str
    models_dir: Path
    allowed_origins: tuple[str, ...]
    gemini_api_key: str | None


def get_settings() -> Settings:
    origins = tuple(origin.strip() for origin in os.getenv("DTI_ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if origin.strip())
    models_dir = Path(os.getenv("DTI_MODELS_DIR", str(ROOT_DIR / "models"))).expanduser()
    return Settings(
        api_version=os.getenv("DTI_API_VERSION", "1.0.0"),
        models_dir=models_dir,
        allowed_origins=origins,
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
    )
