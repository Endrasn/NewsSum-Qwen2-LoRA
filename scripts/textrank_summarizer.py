from __future__ import annotations

import argparse
import math
import re
from collections import Counter

import jieba


SENTENCE_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")
TOKEN_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")
STOPWORDS = {
    "的",
    "了",
    "和",
    "与",
    "及",
    "在",
    "是",
    "为",
    "对",
    "中",
    "等",
    "也",
    "将",
    "并",
    "或",
    "一个",
    "一种",
    "进行",
    "通过",
    "实现",
}


def split_sentences(text: str) -> list[str]:
    return [m.group(0).strip() for m in SENTENCE_RE.finditer(text) if m.group(0).strip()]


def tokenize(sentence: str) -> list[str]:
    words = []
    for word in jieba.cut(sentence):
        word = word.strip()
        if not word or word in STOPWORDS:
            continue
        if TOKEN_RE.fullmatch(word):
            words.append(word)
    return words


def similarity(left: list[str], right: list[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = sum((Counter(left) & Counter(right)).values())
    if overlap == 0:
        return 0.0
    return overlap / (math.log(len(left) + 1.0) + math.log(len(right) + 1.0))


def rank_sentences(sentences: list[str], damping: float = 0.85, steps: int = 60) -> list[float]:
    tokens = [tokenize(sentence) for sentence in sentences]
    size = len(sentences)
    weights = [[0.0] * size for _ in range(size)]

    for i in range(size):
        for j in range(i + 1, size):
            score = similarity(tokens[i], tokens[j])
            weights[i][j] = score
            weights[j][i] = score

    scores = [1.0] * size
    for _ in range(steps):
        next_scores = [1.0 - damping] * size
        for i in range(size):
            for j in range(size):
                if i == j or weights[j][i] == 0:
                    continue
                outgoing = sum(weights[j])
                if outgoing:
                    next_scores[i] += damping * weights[j][i] / outgoing * scores[j]
        scores = next_scores

    return scores


def summarize(text: str, ratio: float = 0.25, max_sentences: int = 4) -> str:
    sentences = split_sentences(text)
    if len(sentences) <= max_sentences:
        return "\n".join(sentences)

    keep = max(1, min(max_sentences, math.ceil(len(sentences) * ratio)))
    scores = rank_sentences(sentences)
    selected = sorted(sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:keep])
    return "\n".join(sentences[i] for i in selected)


def main() -> None:
    parser = argparse.ArgumentParser(description="TextRank 中文新闻摘要")
    parser.add_argument("--input", "-i", help="输入文本文件；省略时从标准输入读取")
    parser.add_argument("--ratio", type=float, default=0.25)
    parser.add_argument("--max-sentences", type=int, default=4)
    args = parser.parse_args()

    if args.input:
        with open(args.input, "r", encoding="utf-8") as file:
            text = file.read()
    else:
        text = input("请输入新闻正文：\n")

    print(summarize(text, ratio=args.ratio, max_sentences=args.max_sentences))


if __name__ == "__main__":
    main()
