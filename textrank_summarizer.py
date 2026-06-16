"""
基于 TextRank 的新闻稿总结程序。

运行方式：
1. 启动网页服务：python textrank_summarizer.py --serve
2. 命令行总结：python textrank_summarizer.py news.txt --sentences 5

可选依赖：
pip install jieba
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from threading import Lock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable

from rouge_eval import evaluate

try:
    import jieba
except ImportError:  # 未安装 jieba 时自动使用内置轻量分词。
    jieba = None

# 常见中文停用词。安装 jieba 后会优先使用 jieba 分词，未安装时使用内置轻量分词。
STOP_WORDS = {
    "的",
    "了",
    "和",
    "是",
    "在",
    "有",
    "对",
    "与",
    "及",
    "等",
    "为",
    "将",
    "就",
    "也",
    "都",
    "而",
    "或",
    "一个",
    "一种",
    "记者",
    "表示",
    "相关",
    "进行",
    "通过",
    "根据",
    "目前",
    "同时",
    "进一步",
}

DEFAULT_QWEN_BASE_MODEL = os.getenv("QWEN_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
DEFAULT_QWEN_ADAPTER_PATH = os.getenv(
    "QWEN_ADAPTER_PATH",
    str(Path(__file__).with_name("adapters") / "qwen2.5-7b-lcsts-lora"),
)
DEFAULT_QWEN_DEVICE_MAP = os.getenv("QWEN_DEVICE_MAP", "auto")
DEFAULT_QWEN_TORCH_DTYPE = os.getenv("QWEN_TORCH_DTYPE", "auto")
DEFAULT_QWEN_LOAD_IN_4BIT = os.getenv("QWEN_LOAD_IN_4BIT", "0").lower() in {
    "1",
    "true",
    "yes",
}
MAX_TEXT_LENGTH = 100_000

_qwen_runtime: "LocalQwenRuntime | None" = None
_qwen_runtime_lock = Lock()


def split_sentences(text: str) -> list[str]:
    """按中文新闻常见标点切分句子，并过滤过短内容。"""
    normalized = re.sub(r"\s+", " ", text.replace("\u3000", " ")).strip()
    if not normalized:
        return []

    pieces = re.split(r"(?<=[。！？!?；;])\s*|(?<=\.)\s+", normalized)
    return [sentence.strip() for sentence in pieces if len(sentence.strip()) >= 6]


def fallback_chinese_tokens(text: str) -> list[str]:
    """未安装 jieba 时，用二字片段做轻量中文分词。"""
    if len(text) == 1:
        return [text]
    return [text[index : index + 2] for index in range(len(text) - 1)]


def tokenize(sentence: str) -> list[str]:
    """提取用于相似度计算的词元；中文优先使用 jieba 分词。"""
    tokens: list[str] = []
    for chunk in re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+", sentence.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", chunk):
            if jieba is not None:
                candidates = [word.strip() for word in jieba.lcut(chunk)]
            else:
                candidates = fallback_chinese_tokens(chunk)
        else:
            candidates = [chunk]

        for token in candidates:
            if len(token) > 1 and token not in STOP_WORDS:
                tokens.append(token)
    return tokens


def sentence_similarity(tokens_a: Iterable[str], tokens_b: Iterable[str]) -> float:
    """计算两个句子的词元重叠相似度。"""
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a or not set_b:
        return 0.0
    overlap = len(set_a & set_b)
    return overlap / (math.log(len(set_a) + 1) + math.log(len(set_b) + 1))


def textrank_scores(sentences: list[str], damping: float = 0.85, rounds: int = 80) -> list[float]:
    """构建句子相似度图，并用 PageRank 思路迭代得到句子权重。"""
    tokenized = [tokenize(sentence) for sentence in sentences]
    size = len(sentences)
    if size == 0:
        return []
    if size == 1:
        return [1.0]

    graph = [[0.0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            score = sentence_similarity(tokenized[i], tokenized[j])
            graph[i][j] = score
            graph[j][i] = score

    weights = [1.0 for _ in range(size)]
    out_sums = [sum(row) for row in graph]

    for _ in range(rounds):
        next_weights = [1.0 - damping for _ in range(size)]
        for i in range(size):
            for j in range(size):
                if i == j or graph[j][i] == 0 or out_sums[j] == 0:
                    continue
                next_weights[i] += damping * graph[j][i] / out_sums[j] * weights[j]
        weights = next_weights

    return weights


def summarize(text: str, max_sentences: int = 5, ratio: float = 0.3) -> str:
    """生成摘要：按 TextRank 得分选句，再按原文顺序输出，保证可读性。"""
    sentences = split_sentences(text)
    if not sentences:
        return ""

    count_by_ratio = max(1, round(len(sentences) * ratio))
    target_count = min(max_sentences, count_by_ratio, len(sentences))
    scores = textrank_scores(sentences)
    ranked_indexes = sorted(range(len(sentences)), key=lambda index: scores[index], reverse=True)
    selected_indexes = sorted(ranked_indexes[:target_count])
    return "\n".join(sentences[index] for index in selected_indexes)


class LocalQwenRuntime:
    """Load Qwen base weights plus the LoRA adapter once, then reuse them."""

    def __init__(
        self,
        base_model: str,
        adapter_path: str,
        device_map: str = DEFAULT_QWEN_DEVICE_MAP,
        torch_dtype: str = DEFAULT_QWEN_TORCH_DTYPE,
        load_in_4bit: bool = DEFAULT_QWEN_LOAD_IN_4BIT,
    ) -> None:
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.load_in_4bit = load_in_4bit
        self.tokenizer: Any = None
        self.model: Any = None
        self.torch: Any = None
        self._load_lock = Lock()

    def load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        with self._load_lock:
            if self.model is not None and self.tokenizer is not None:
                return

            try:
                import torch
                from peft import PeftModel
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "本地 Qwen LoRA 推理需要安装 torch、transformers、peft、accelerate。"
                ) from exc

            model_kwargs: dict[str, Any] = {
                "device_map": self.device_map,
                "trust_remote_code": True,
            }
            if self.torch_dtype == "auto":
                model_kwargs["torch_dtype"] = "auto"
            else:
                model_kwargs["torch_dtype"] = getattr(torch, self.torch_dtype)

            if self.load_in_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                except ImportError as exc:
                    raise RuntimeError("启用 4-bit 推理需要安装 bitsandbytes。") from exc
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True
                )

            tokenizer_source = (
                self.adapter_path if Path(self.adapter_path).exists() else self.base_model
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source,
                trust_remote_code=True,
            )
            base = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                **model_kwargs,
            )
            self.model = PeftModel.from_pretrained(base, self.adapter_path)
            self.model.eval()
            self.torch = torch

    def generate(self, text: str, max_tokens: int = 256) -> str:
        self.load()

        messages = [
            {
                "role": "system",
                "content": "你是严谨的中文新闻编辑，只输出简洁、忠实原文的新闻摘要。",
            },
            {
                "role": "user",
                "content": (
                    "请为下面的中文新闻稿生成一段准确、简洁的摘要。"
                    "保留关键人物、事件、时间和结果，不要添加原文没有的信息。\n\n"
                    f"新闻稿：\n{text}"
                ),
            },
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)

        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max(32, min(max_tokens, 1024)),
                temperature=0.2,
                top_p=0.9,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
        summary = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()
        if not summary:
            raise RuntimeError("Qwen LoRA 模型返回了空摘要")
        return summary


def get_qwen_runtime(
    base_model: str = DEFAULT_QWEN_BASE_MODEL,
    adapter_path: str = DEFAULT_QWEN_ADAPTER_PATH,
    load_in_4bit: bool = DEFAULT_QWEN_LOAD_IN_4BIT,
) -> LocalQwenRuntime:
    global _qwen_runtime

    with _qwen_runtime_lock:
        if (
            _qwen_runtime is None
            or _qwen_runtime.base_model != base_model
            or _qwen_runtime.adapter_path != adapter_path
            or _qwen_runtime.load_in_4bit != load_in_4bit
        ):
            _qwen_runtime = LocalQwenRuntime(
                base_model=base_model,
                adapter_path=adapter_path,
                load_in_4bit=load_in_4bit,
            )
        return _qwen_runtime


def generate_qwen_summary(
    text: str,
    base_model: str = DEFAULT_QWEN_BASE_MODEL,
    adapter_path: str = DEFAULT_QWEN_ADAPTER_PATH,
    max_tokens: int = 256,
    load_in_4bit: bool = DEFAULT_QWEN_LOAD_IN_4BIT,
) -> str:
    runtime = get_qwen_runtime(
        base_model=base_model,
        adapter_path=adapter_path,
        load_in_4bit=load_in_4bit,
    )
    return runtime.generate(text, max_tokens=max_tokens)

class SummaryHandler(BaseHTTPRequestHandler):
    """为 HTML 页面提供静态文件和摘要接口。"""

    server_version = "TextRankSummaryServer/1.0"

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            html_path = Path(__file__).with_name("index.html")
            self._send_bytes(html_path.read_bytes(), "text/html; charset=utf-8")
            return
        if self.path == "/rouge.html":
            html_path = Path(__file__).with_name("rouge.html")
            self._send_bytes(html_path.read_bytes(), "text/html; charset=utf-8")
            return
        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        if self.path == "/analyze":
            self._handle_analyze()
            return

        if self.path == "/rouge":
            self._handle_rouge()
            return

        if self.path != "/summarize":
            self.send_error(404, "Not found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            text = str(payload.get("text", ""))
            max_sentences = int(payload.get("max_sentences", 5))
            ratio = float(payload.get("ratio", 0.3))
            summary = summarize(text, max_sentences=max_sentences, ratio=ratio)
            body = json.dumps({"summary": summary}, ensure_ascii=False).encode("utf-8")
            self._send_bytes(body, "application/json; charset=utf-8")
        except Exception as exc:  # noqa: BLE001 - 返回给网页一个明确错误。
            body = json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8")
            self._send_bytes(body, "application/json; charset=utf-8", status=400)

    def _handle_analyze(self) -> None:
        """生成 TextRank 与 Qwen 摘要，并按需计算两者的 ROUGE 分数。"""
        try:
            payload = self._read_json()
            text = str(payload.get("text", "")).strip()
            reference = str(payload.get("reference", "")).strip()
            if not text:
                raise ValueError("请输入新闻稿")
            if len(text) > MAX_TEXT_LENGTH:
                raise ValueError(f"新闻稿不能超过 {MAX_TEXT_LENGTH} 个字符")

            max_sentences = max(1, min(int(payload.get("max_sentences", 5)), 20))
            ratio = max(0.05, min(float(payload.get("ratio", 0.3)), 1.0))
            mode = str(payload.get("mode", "char"))
            if mode not in {"char", "word"}:
                mode = "char"

            textrank_summary = summarize(
                text,
                max_sentences=max_sentences,
                ratio=ratio,
            )
            result: dict[str, object] = {
                "textrank": {
                    "summary": textrank_summary,
                    "scores": evaluate(textrank_summary, reference, mode)
                    if reference and textrank_summary
                    else None,
                },
                "qwen": {
                    "summary": "",
                    "scores": None,
                    "error": None,
                },
            }

            try:
                load_in_4bit_value = payload.get(
                    "load_in_4bit",
                    DEFAULT_QWEN_LOAD_IN_4BIT,
                )
                if isinstance(load_in_4bit_value, str):
                    load_in_4bit = load_in_4bit_value.lower() in {
                        "1",
                        "true",
                        "yes",
                    }
                else:
                    load_in_4bit = bool(load_in_4bit_value)

                qwen_summary = generate_qwen_summary(
                    text,
                    base_model=str(
                        payload.get("base_model") or DEFAULT_QWEN_BASE_MODEL
                    ),
                    adapter_path=str(
                        payload.get("adapter_path") or DEFAULT_QWEN_ADAPTER_PATH
                    ),
                    max_tokens=int(payload.get("max_tokens", 256)),
                    load_in_4bit=load_in_4bit,
                )
                result["qwen"] = {
                    "summary": qwen_summary,
                    "scores": evaluate(qwen_summary, reference, mode)
                    if reference
                    else None,
                    "error": None,
                }
            except Exception as exc:  # noqa: BLE001 - TextRank 结果仍然可用。
                result["qwen"] = {
                    "summary": "",
                    "scores": None,
                    "error": str(exc),
                }

            body = json.dumps(result, ensure_ascii=False).encode("utf-8")
            self._send_bytes(body, "application/json; charset=utf-8")
        except Exception as exc:  # noqa: BLE001 - 返回明确的参数错误。
            body = json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8")
            self._send_bytes(body, "application/json; charset=utf-8", status=400)

    def _handle_rouge(self) -> None:
        """接收网页传入的摘要和对照文本，返回 ROUGE 评分。"""
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            candidate = str(payload.get("candidate", ""))
            reference = str(payload.get("reference", ""))
            mode = str(payload.get("mode", "char"))
            if mode not in {"char", "word"}:
                mode = "char"
            scores = evaluate(candidate, reference, mode=mode)
            body = json.dumps({"scores": scores}, ensure_ascii=False).encode("utf-8")
            self._send_bytes(body, "application/json; charset=utf-8")
        except Exception as exc:  # noqa: BLE001 - 返回给网页一个明确错误。
            body = json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8")
            self._send_bytes(body, "application/json; charset=utf-8", status=400)

    def _read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            raise ValueError("请求内容为空")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("请求内容必须是 JSON 对象")
        return payload

    def log_message(self, format: str, *args: object) -> None:
        """减少控制台噪声，只保留启动信息和错误。"""
        return

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


def serve(host: str = "127.0.0.1", port: int = 8080) -> None:
    """启动本地网页服务。"""
    server = ThreadingHTTPServer((host, port), SummaryHandler)
    print(f"新闻稿总结网页已启动：http://{host}:{port}")
    print("按 Ctrl+C 停止服务。")
    server.serve_forever()


def build_parser() -> argparse.ArgumentParser:
    """定义命令行参数，兼容网页模式和文件模式。"""
    parser = argparse.ArgumentParser(description="基于 TextRank 的新闻稿总结程序")
    parser.add_argument("file", nargs="?", help="需要总结的 txt 文件路径")
    parser.add_argument("--serve", action="store_true", help="启动网页服务")
    parser.add_argument("--host", default="127.0.0.1", help="网页服务监听地址")
    parser.add_argument("--port", type=int, default=8080, help="网页服务端口")
    parser.add_argument("--sentences", type=int, default=5, help="摘要最多保留句子数")
    parser.add_argument("--ratio", type=float, default=0.3, help="摘要句子占原文比例")
    return parser


def main() -> None:
    """程序入口。"""
    args = build_parser().parse_args()
    if args.serve or not args.file:
        serve(args.host, args.port)
        return

    text = Path(args.file).read_text(encoding="utf-8")
    print(summarize(text, max_sentences=args.sentences, ratio=args.ratio))


if __name__ == "__main__":
    main()
