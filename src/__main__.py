"""Command-line interface for the RAG pipeline built with Python Fire."""

import json
from pathlib import Path
import sys

import fire
from pydantic import ValidationError
from tqdm import tqdm

from .models import (
    MinimalAnswer,
    MinimalSearchResults,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from .rag.evaluator import evaluate_search_results
from .rag.generator import AnswerGenerator, DEFAULT_MODEL_NAME
from .rag.indexer import build_index
from .rag.retriever import BM25Retriever
from .rag.api import run_server


class RagCLI:
    """CLI exposing indexing, retrieval, generation, and evaluation."""

    def index(
        self,
        max_chunk_size: int = 2000,
        raw_dir: str = "data/raw",
        output_dir: str = "data/processed",
        overlap: int = 200,
    ) -> None:
        """Ingest raw_dir and build the BM25 index under output_dir."""
        try:
            num_chunks = build_index(
                raw_dir=raw_dir,
                output_dir=output_dir,
                max_chunk_size=max_chunk_size,
                overlap=overlap,
            )
            print(
                f"Ingestion complete! Indexed {num_chunks} chunks "
                f"under {output_dir}/"
            )
        except (ValueError, FileNotFoundError, OSError) as exc:
            print(f"Error during indexing: {exc}", file=sys.stderr)

    def search(
        self,
        query: str = "",
        k: int = 5,
        index_dir: str = "data/processed",
    ) -> None:
        """Return top-k retrieved sources for a single query."""
        try:
            query_str = "" if query is None else str(query)
            if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
                print("No results (k must be a positive integer).")
                return
            if not query_str.strip():
                print("No results (empty query).")
                return

            retriever = BM25Retriever(index_dir=index_dir)
            sources = retriever.search(query_str, k=k)
            if not sources:
                print("No matching sources found.")
                return
            for src in sources:
                span = (
                    f"[{src.first_character_index}:"
                    f"{src.last_character_index}]"
                )
                print(f"{src.file_path} {span}")
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"Error during search: {exc}", file=sys.stderr)

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
        index_dir: str = "data/processed",
    ) -> None:
        """Run search over a JSON dataset and save StudentSearchResults."""
        try:
            if isinstance(k, bool) or not isinstance(k, int) or k < 0:
                raise ValueError("k must be a non-negative integer.")

            ds_file = Path(dataset_path)
            with ds_file.open("r", encoding="utf-8") as handle:
                dataset = RagDataset.model_validate(json.load(handle))

            retriever = BM25Retriever(index_dir=index_dir)
            results: list[MinimalSearchResults] = []
            for item in tqdm(
                dataset.rag_questions, desc="Searching", unit="q"
            ):
                sources = retriever.search(item.question, k=k)
                results.append(
                    MinimalSearchResults(
                        question_id=item.question_id,
                        question=item.question,
                        retrieved_sources=sources,
                    )
                )

            payload = StudentSearchResults(search_results=results, k=k)
            out_dir = Path(save_directory)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / ds_file.name
            with out_file.open("w", encoding="utf-8") as handle:
                handle.write(payload.model_dump_json(indent=2))
            print(f"Saved student_search_results to {out_file}")
        except (
            FileNotFoundError,
            json.JSONDecodeError,
            ValidationError,
            ValueError,
            OSError,
        ) as exc:
            print(f"Error during search_dataset: {exc}", file=sys.stderr)

    def answer(
        self,
        query: str = "",
        k: int = 5,
        index_dir: str = "data/processed",
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        """Answer a single query using retrieved snippets and Qwen3-0.6B."""
        try:
            query_str = "" if query is None else str(query)
            if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
                print("No answer (k must be a positive integer).")
                return
            if not query_str.strip():
                print("No answer (empty query).")
                return

            retriever = BM25Retriever(index_dir=index_dir)
            sources = retriever.search(query_str, k=k)
            generator = AnswerGenerator(model_name=model_name)
            text = generator.generate(query_str, sources)
            print(text)
        except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
            print(f"Error during answer: {exc}", file=sys.stderr)

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        """Generate grounded answers for a StudentSearchResults JSON file."""
        try:
            in_file = Path(student_search_results_path)
            with in_file.open("r", encoding="utf-8") as handle:
                search_data = StudentSearchResults.model_validate(
                    json.load(handle)
                )

            generator = AnswerGenerator(model_name=model_name)
            answered: list[MinimalAnswer] = []
            for item in tqdm(
                search_data.search_results, desc="Answering", unit="q"
            ):
                text = generator.generate(
                    item.question, item.retrieved_sources
                )
                answered.append(
                    MinimalAnswer(
                        question_id=item.question_id,
                        question=item.question,
                        retrieved_sources=item.retrieved_sources,
                        answer=text,
                    )
                )

            payload = StudentSearchResultsAndAnswer(
                search_results=answered, k=search_data.k
            )
            out_dir = Path(save_directory)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / in_file.name
            with out_file.open("w", encoding="utf-8") as handle:
                handle.write(payload.model_dump_json(indent=2))
            print(f"Saved student_search_results_and_answer to {out_file}")
        except (
            FileNotFoundError,
            json.JSONDecodeError,
            ValidationError,
            ValueError,
            OSError,
            RuntimeError,
        ) as exc:
            print(f"Error during answer_dataset: {exc}", file=sys.stderr)

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
    ) -> None:
        """Report local Recall@k against a ground-truth dataset."""
        try:
            recalls = evaluate_search_results(
                student_results_path=student_search_results_path,
                dataset_path=dataset_path,
            )
            summary = " ".join(
                f"Recall@{k}: {score:.3f} ({score * 100:.1f}%)"
                for k, score in recalls.items()
            )
            print(summary)
        except (
            FileNotFoundError,
            json.JSONDecodeError,
            ValidationError,
            ValueError,
            OSError,
        ) as exc:
            print(f"Error during evaluate: {exc}", file=sys.stderr)

    def serve(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        index_dir: str = "data/processed",
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        """Start the local HTTP API server for search and answer queries."""
        try:
            run_server(
                host=host,
                port=port,
                index_dir=index_dir,
                model_name=model_name,
            )
        except (FileNotFoundError, OSError, ValueError) as exc:
            print(f"Error starting HTTP API: {exc}", file=sys.stderr)


def main() -> None:
    """Entry point for ``python -m src``."""
    fire.Fire(RagCLI)


if __name__ == "__main__":
    main()
