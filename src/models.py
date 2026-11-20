"""Pydantic data models for the RAG pipeline."""

from typing import List
import uuid

from pydantic import BaseModel, Field, model_validator


class MinimalSource(BaseModel):
    """Represents a character span inside a single source file."""

    file_path: str
    first_character_index: int
    last_character_index: int

    @model_validator(mode="after")
    def validate_span(self) -> "MinimalSource":
        """Ensure character indices form a valid non-negative range."""
        if self.first_character_index < 0:
            raise ValueError("first_character_index must be >= 0")
        if self.last_character_index < self.first_character_index:
            raise ValueError(
                "last_character_index must be >= first_character_index"
            )
        return self


class UnansweredQuestion(BaseModel):
    """Represents a question without ground-truth sources or answer."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represents a question with ground-truth sources and answer."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Represents a dataset of answered or unanswered RAG questions."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Represents retrieved sources for a single question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Represents retrieved sources together with a generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Output format for batch search results."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Output format for batch search results with generated answers."""

    search_results: List[MinimalAnswer]
    k: int
