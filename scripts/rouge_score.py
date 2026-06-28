from __future__ import annotations

import argparse
import json

import jieba
from rouge_chinese import Rouge


def _spaced(text: str) -> str:
    return " ".join(jieba.cut(text))


def score_summary(reference: str, hypothesis: str) -> dict[str, dict[str, float]]:
    rouge = Rouge()
    scores = rouge.get_scores(_spaced(hypothesis), _spaced(reference))[0]
    return {
        name: {
            "precision": round(values["p"], 4),
            "recall": round(values["r"], 4),
            "f1": round(values["f"], 4),
        }
        for name, values in scores.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="计算中文摘要 ROUGE-1/2/L")
    parser.add_argument("--reference", "-r", required=True, help="参考摘要文本")
    parser.add_argument("--hypothesis", "-p", required=True, help="待评估摘要文本")
    args = parser.parse_args()

    print(json.dumps(score_summary(args.reference, args.hypothesis), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
