"""
ROUGE-1、ROUGE-2、ROUGE-L 检测脚本。

用法示例：
python rouge_eval.py
python rouge_eval.py system_summary.txt reference_summary.txt
python rouge_eval.py system_summary.txt reference_summary.txt --mode char --json

说明：
- 不带参数运行时，会启动网页评分服务。
- system_summary.txt：程序生成的摘要。
- reference_summary.txt：人工写好的参考摘要。
- 评分计算使用项目内置实现。
- 默认使用 char 模式，更适合没有分词工具的中文文本。
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def read_text_with_fallback(path: str) -> str:
    """读取文本文件；优先 UTF-8，失败时兼容常见中文 Windows 编码。"""
    data = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


class RougeTokenizer:
    """兼容旧调用方式的分词器。"""

    def __init__(self, mode: str = "char") -> None:
        self.mode = mode

    def tokenize(self, text: str) -> list[str]:
        return tokenize(text, self.mode)


def tokenize(text: str, mode: str = "char") -> list[str]:
    """把文本切成 ROUGE 计算用的 token。"""
    text = text.lower()

    if mode == "word":
        # 英文按词，中文连续汉字会作为一个词；适合已经用空格分词的文本。
        return re.findall(r"[\u4e00-\u9fff]+|[a-z0-9]+", text)

    # char 模式：中文按单字，英文和数字按词，忽略标点空白。
    tokens: list[str] = []
    for chunk in re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", text):
        tokens.append(chunk)
    return tokens


def _prf(overlap: int, candidate_count: int, reference_count: int) -> dict[str, float]:
    precision = overlap / candidate_count if candidate_count else 0.0
    recall = overlap / reference_count if reference_count else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _ngram_score(
    candidate_tokens: list[str],
    reference_tokens: list[str],
    size: int,
) -> dict[str, float]:
    candidate_ngrams = Counter(
        tuple(candidate_tokens[index : index + size])
        for index in range(len(candidate_tokens) - size + 1)
    )
    reference_ngrams = Counter(
        tuple(reference_tokens[index : index + size])
        for index in range(len(reference_tokens) - size + 1)
    )
    overlap = sum((candidate_ngrams & reference_ngrams).values())
    return _prf(overlap, sum(candidate_ngrams.values()), sum(reference_ngrams.values()))


def _lcs_length(left: list[str], right: list[str]) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = [0] * (len(left) + 1)
    for right_token in right:
        current = [0]
        for index, left_token in enumerate(left, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def evaluate(candidate: str, reference: str, mode: str = "char") -> dict[str, dict[str, float]]:
    """计算 ROUGE-1、ROUGE-2 和基于最长公共子序列的 ROUGE-L。"""
    candidate_tokens = tokenize(candidate, mode)
    reference_tokens = tokenize(reference, mode)
    lcs = _lcs_length(candidate_tokens, reference_tokens)
    return {
        "rouge-1": _ngram_score(candidate_tokens, reference_tokens, 1),
        "rouge-2": _ngram_score(candidate_tokens, reference_tokens, 2),
        "rouge-l": _prf(lcs, len(candidate_tokens), len(reference_tokens)),
    }


def format_score(value: float) -> str:
    """把小数格式化为百分比字符串。"""
    return f"{value * 100:.2f}%"


def print_table(scores: dict[str, dict[str, float]]) -> None:
    """以表格形式输出结果。"""
    print("Metric    Precision  Recall     F1")
    print("--------------------------------------")
    for metric, values in scores.items():
        print(
            f"{metric:<9} "
            f"{format_score(values['precision']):>9}  "
            f"{format_score(values['recall']):>8}  "
            f"{format_score(values['f1']):>8}"
        )


class RougeHandler(BaseHTTPRequestHandler):
    """为 rouge.html 提供页面和 ROUGE 评分接口。"""

    server_version = "RougeEvalServer/1.0"

    def do_GET(self) -> None:
        """返回 ROUGE 网页。"""
        if self.path in {"/", "/rouge.html"}:
            html_path = Path(__file__).with_name("rouge.html")
            if not html_path.exists():
                self.send_error(404, "rouge.html not found")
                return
            self._send_bytes(html_path.read_bytes(), "text/html; charset=utf-8")
            return
        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        """接收网页传入的文本并返回 ROUGE 分数。"""
        if self.path != "/rouge":
            self.send_error(404, "Not found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            candidate = str(payload.get("candidate", ""))
            reference = str(payload.get("reference", ""))
            mode = str(payload.get("mode", "char"))
            if mode not in {"char", "word"}:
                mode = "char"
            body = json.dumps(
                {"scores": evaluate(candidate, reference, mode=mode)},
                ensure_ascii=False,
            ).encode("utf-8")
            self._send_bytes(body, "application/json; charset=utf-8")
        except Exception as exc:  # noqa: BLE001 - 返回给网页明确错误信息。
            body = json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8")
            self._send_bytes(body, "application/json; charset=utf-8", status=400)

    def log_message(self, format: str, *args: object) -> None:
        """减少控制台访问日志噪声。"""
        return

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        """发送 HTTP 响应。"""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


def serve(host: str = "127.0.0.1", port: int = 8001) -> None:
    """启动 ROUGE 网页评分服务。"""
    server = ThreadingHTTPServer((host, port), RougeHandler)
    print(f"ROUGE 网页评分服务已启动：http://{host}:{port}/rouge.html")
    print("在网页中输入大模型总结和对照文本即可计算得分。")
    print("按 Ctrl+C 停止服务。")
    server.serve_forever()


def build_parser() -> argparse.ArgumentParser:
    """定义命令行参数。"""
    parser = argparse.ArgumentParser(description="检测摘要文本的 ROUGE-1、ROUGE-2、ROUGE-L")
    parser.add_argument("candidate", nargs="?", help="系统生成摘要 txt 文件")
    parser.add_argument("reference", nargs="?", help="人工参考摘要 txt 文件")
    parser.add_argument(
        "--mode",
        choices=("char", "word"),
        default="char",
        help="分词模式：char 适合中文，word 适合空格分词文本",
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON 格式结果")
    parser.add_argument("--serve", action="store_true", help="启动网页评分服务")
    parser.add_argument("--host", default="127.0.0.1", help="网页服务监听地址")
    parser.add_argument("--port", type=int, default=8001, help="网页服务端口")
    return parser


def main() -> None:
    """程序入口。"""
    parser = build_parser()
    args = parser.parse_args()

    if args.serve or (not args.candidate and not args.reference):
        serve(args.host, args.port)
        return

    if not args.candidate or not args.reference:
        parser.error("命令行评分需要同时提供 candidate 和 reference；网页评分请直接运行 python rouge_eval.py")

    candidate = read_text_with_fallback(args.candidate)
    reference = read_text_with_fallback(args.reference)
    scores = evaluate(candidate, reference, args.mode)

    if args.json:
        print(json.dumps(scores, ensure_ascii=False, indent=2))
    else:
        print_table(scores)


if __name__ == "__main__":
    main()
