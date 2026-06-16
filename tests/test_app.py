from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
from urllib.request import Request, urlopen

from rouge_eval import evaluate
from textrank_summarizer import SummaryHandler, split_sentences, summarize


NEWS = (
    "某市今日发布人工智能产业发展计划。"
    "计划提出建设公共算力平台，为中小企业提供技术支持。"
    "当地还将设立专项基金，推动人工智能在医疗和教育领域应用。"
    "相关部门表示，项目实施过程中将同步加强数据安全管理。"
)
REFERENCE = "某市发布人工智能产业计划，将建设算力平台并加强数据安全。"
QWEN_SUMMARY = "某市发布人工智能产业计划，建设算力平台并推动行业应用。"


class ServerFixture:
    def __init__(self, handler: type[BaseHTTPRequestHandler]) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    def __enter__(self) -> "ServerFixture":
        self.thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class SummaryTests(unittest.TestCase):
    def test_rouge_scores_identical_text_as_one(self) -> None:
        scores = evaluate(REFERENCE, REFERENCE)
        for values in scores.values():
            self.assertEqual(values["f1"], 1.0)

    def test_textrank_summary_respects_sentence_count(self) -> None:
        self.assertEqual(len(split_sentences(NEWS)), 4)
        result = summarize(NEWS, max_sentences=2, ratio=0.5)
        self.assertEqual(len(result.splitlines()), 2)

    def test_analyze_endpoint_returns_both_summaries_and_scores(self) -> None:
        with patch(
            "textrank_summarizer.generate_qwen_summary",
            return_value=QWEN_SUMMARY,
        ) as mock_generate:
            with ServerFixture(SummaryHandler) as app:
                payload = {
                    "text": NEWS,
                    "reference": REFERENCE,
                    "base_model": "test-base-model",
                    "adapter_path": "test-adapter-path",
                    "load_in_4bit": True,
                    "ratio": 0.5,
                    "max_sentences": 2,
                }
                request = Request(
                    f"{app.url}/analyze",
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=30) as response:
                    result = json.loads(response.read().decode("utf-8"))

        mock_generate.assert_called_once()
        self.assertEqual(mock_generate.call_args.kwargs["base_model"], "test-base-model")
        self.assertEqual(
            mock_generate.call_args.kwargs["adapter_path"],
            "test-adapter-path",
        )
        self.assertTrue(mock_generate.call_args.kwargs["load_in_4bit"])
        self.assertTrue(result["textrank"]["summary"])
        self.assertIn("rouge-1", result["textrank"]["scores"])
        self.assertEqual(result["qwen"]["summary"], QWEN_SUMMARY)
        self.assertIn("rouge-l", result["qwen"]["scores"])
        self.assertIsNone(result["qwen"]["error"])


if __name__ == "__main__":
    unittest.main()
