"""
Central configuration for MF-Guide RAG Chatbot.

All modules import Config from here. No hardcoded values elsewhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

# Load .env file at module level so all imports get env vars
load_dotenv()


# ---------------------------------------------------------------------------
# Constants (not configurable via env — fixed for the demo)
# ---------------------------------------------------------------------------

GROWW_URLS: List[str] = [
    "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
    "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth",
    "https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
    "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
    "https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth",
]

SCHEME_MAP: Dict[str, Dict[str, str]] = {
    "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth": {
        "scheme": "HDFC Large Cap Fund - Direct Growth",
        "category": "Large Cap",
    },
    "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth": {
        "scheme": "HDFC Equity Fund - Direct Growth",
        "category": "Flexi Cap",
    },
    "https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth": {
        "scheme": "HDFC ELSS Tax Saver Fund - Direct Plan Growth",
        "category": "ELSS",
    },
    "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth": {
        "scheme": "HDFC Small Cap Fund - Direct Growth",
        "category": "Small Cap",
    },
    "https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth": {
        "scheme": "HDFC Balanced Advantage Fund - Direct Growth",
        "category": "Balanced Advantage",
    },
}


# ---------------------------------------------------------------------------
# Config Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """
    Immutable configuration for the MF-Guide RAG pipeline.

    Use Config.from_env() to create an instance with values from environment
    variables (with sensible defaults).
    """

    # --- Data paths ---
    BASE_DIR: Path = field(default_factory=lambda: Path(__file__).parent)
    RAW_DATA_DIR: Path = field(init=False)
    VECTOR_STORE_DIR: Path = field(init=False)

    # --- Source URLs ---
    GROWW_URLS: List[str] = field(default_factory=lambda: GROWW_URLS)
    SCHEME_MAP: Dict[str, Dict[str, str]] = field(default_factory=lambda: SCHEME_MAP)

    # --- Chunking ---
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100

    # --- Embedding ---
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Retrieval ---
    TOP_K: int = 5
    SCORE_THRESHOLD: float = 0.3
    MEMORY_WINDOW: int = 10

    # --- LLM ---
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 500

    # --- API Keys ---
    GROQ_API_KEY: str = ""

    def __post_init__(self):
        # frozen=True means we need object.__setattr__ for derived fields
        object.__setattr__(self, "RAW_DATA_DIR", self.BASE_DIR / "data" / "raw_pages")
        object.__setattr__(self, "VECTOR_STORE_DIR", self.BASE_DIR / "vector_store")

    # -----------------------------------------------------------------------
    # Factory
    # -----------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "Config":
        """
        Build a Config instance, reading overrides from environment variables.

        Supported env vars:
            OPENAI_API_KEY         — OpenAI API key (required for LLM)
            CHUNK_SIZE             — override chunk size (default 500)
            CHUNK_OVERLAP          — override chunk overlap (default 100)
            EMBEDDING_MODEL        — override embedding model
            TOP_K                  — override retrieval top-k (default 5)
            SCORE_THRESHOLD        — override similarity threshold (default 0.3)
            MEMORY_WINDOW          — override conversation memory window (default 10)
            LLM_MODEL              — override LLM model name
            LLM_TEMPERATURE        — override LLM temperature
            LLM_MAX_TOKENS         — override max output tokens
        """
        return cls(
            GROQ_API_KEY=os.getenv("GROQ_API_KEY", ""),
            CHUNK_SIZE=int(os.getenv("CHUNK_SIZE", "500")),
            CHUNK_OVERLAP=int(os.getenv("CHUNK_OVERLAP", "100")),
            EMBEDDING_MODEL=os.getenv(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            TOP_K=int(os.getenv("TOP_K", "5")),
            SCORE_THRESHOLD=float(os.getenv("SCORE_THRESHOLD", "0.3")),
            MEMORY_WINDOW=int(os.getenv("MEMORY_WINDOW", "10")),
            LLM_MODEL=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
            LLM_TEMPERATURE=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            LLM_MAX_TOKENS=int(os.getenv("LLM_MAX_TOKENS", "500")),
        )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def validate(self) -> List[str]:
        """
        Return a list of validation error messages.
        Empty list means config is valid.
        """
        errors: List[str] = []
        if not self.GROQ_API_KEY:
            errors.append("GROQ_API_KEY is not set. Add it to .env file.")
        if self.CHUNK_SIZE < 100:
            errors.append(f"CHUNK_SIZE ({self.CHUNK_SIZE}) is too small. Minimum 100.")
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            errors.append(
                f"CHUNK_OVERLAP ({self.CHUNK_OVERLAP}) must be < CHUNK_SIZE ({self.CHUNK_SIZE})."
            )
        if self.TOP_K < 1:
            errors.append(f"TOP_K ({self.TOP_K}) must be >= 1.")
        if self.MEMORY_WINDOW < 0:
            errors.append(f"MEMORY_WINDOW ({self.MEMORY_WINDOW}) must be >= 0.")
        if not (0.0 <= self.SCORE_THRESHOLD <= 1.0):
            errors.append(
                f"SCORE_THRESHOLD ({self.SCORE_THRESHOLD}) must be between 0.0 and 1.0."
            )
        if self.LLM_TEMPERATURE < 0.0 or self.LLM_TEMPERATURE > 2.0:
            errors.append(
                f"LLM_TEMPERATURE ({self.LLM_TEMPERATURE}) must be between 0.0 and 2.0."
            )
        return errors
