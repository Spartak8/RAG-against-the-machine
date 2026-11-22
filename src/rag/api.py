"""Minimal local HTTP API for querying the RAG index and answering."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from typing import Any

from .generator import AnswerGenerator, DEFAULT_MODEL_NAME
from .retriever import BM25Retriever


class RAGServerState:
    """Shared state holding the cached retriever and lazy-loaded LLM."""

    def __init__(
        self,
        index_dir: str = "data/processed",
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        """Initialize the retriever; defer LLM loading until first /answer."""
        self.retriever = BM25Retriever(index_dir=index_dir)
        self.model_name = model_name
        self._generator: AnswerGenerator | None = None

    def get_generator(self) -> AnswerGenerator:
        """Return the cached AnswerGenerator, loading it on first use."""
        if self._generator is None:
            self._generator = AnswerGenerator(model_name=self.model_name)
        return self._generator


def _make_handler(
    state: RAGServerState,
) -> type[BaseHTTPRequestHandler]:
    """Build a request handler class bound to the given RAGServerState."""

    class RAGRequestHandler(BaseHTTPRequestHandler):
        """HTTP handler exposing /health, /search, and /answer endpoints."""

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            """Serialize payload as JSON and write the HTTP response."""
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self) -> dict[str, Any] | None:
            """Parse the request body as a JSON object, or return None."""
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return None
            raw = self.rfile.read(max(0, length))
            try:
                parsed = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return None
            return parsed if isinstance(parsed, dict) else None

        def do_GET(self) -> None:
            """Handle GET /health."""
            if self.path == "/health":
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "indexed_chunks": len(state.retriever.sources),
                    },
                )
                return
            self._send_json(404, {"error": "Not found"})

        def do_POST(self) -> None:
            """Handle POST /search and POST /answer."""
            if self.path not in ("/search", "/answer"):
                self._send_json(404, {"error": "Not found"})
                return

            body = self._read_json_body()
            if body is None:
                self._send_json(400, {"error": "Invalid JSON body"})
                return

            query = body.get("query", "")
            k = body.get("k", 5)
            if not isinstance(query, str) or not query.strip():
                self._send_json(400, {"error": "Field 'query' is required"})
                return
            if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
                self._send_json(
                    400, {"error": "Field 'k' must be a positive integer"}
                )
                return

            sources = state.retriever.search(query, k=k)
            response: dict[str, Any] = {
                "query": query,
                "k": k,
                "retrieved_sources": [s.model_dump() for s in sources],
            }

            if self.path == "/answer":
                try:
                    generator = state.get_generator()
                    response["answer"] = generator.generate(query, sources)
                except (OSError, RuntimeError, ValueError) as exc:
                    self._send_json(500, {"error": str(exc)})
                    return

            self._send_json(200, response)

    return RAGRequestHandler


def create_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    index_dir: str = "data/processed",
    model_name: str = DEFAULT_MODEL_NAME,
) -> ThreadingHTTPServer:
    """Create a configured ThreadingHTTPServer for the RAG API."""
    state = RAGServerState(index_dir=index_dir, model_name=model_name)
    handler_cls = _make_handler(state)
    return ThreadingHTTPServer((host, port), handler_cls)


def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    index_dir: str = "data/processed",
    model_name: str = DEFAULT_MODEL_NAME,
) -> None:
    """Start the local RAG HTTP server and block until interrupted."""
    server = create_server(
        host=host, port=port, index_dir=index_dir, model_name=model_name
    )
    print(f"RAG HTTP API listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down RAG HTTP API.")
    finally:
        server.server_close()
