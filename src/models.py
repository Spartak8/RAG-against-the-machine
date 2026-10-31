"""Pydantic data models exchanged between the RAG pipeline stages."""

import uuid
from typing import List

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """A location in the corpus: a file and a character range."""

    file_path: str
    first_character_index: int
    last_character_index: int
