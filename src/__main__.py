import fire


class RagCLI:
    """Command-line interface for the RAG pipeline."""

    def hello(self, name: str = "world") -> str:
        """Print a greeting (temporary smoke test)."""
        return f"hello, {name}"


if __name__ == "__main__":
    fire.Fire(RagCLI)
