from __future__ import annotations

from flask import Flask, jsonify, request, send_from_directory

from scripts.rouge_score import score_summary
from scripts.textrank_summarizer import summarize as textrank_summarize


app = Flask(__name__, static_folder="static", static_url_path="")


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.post("/api/textrank")
def api_textrank():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    ratio = float(payload.get("ratio") or 0.25)
    max_sentences = int(payload.get("max_sentences") or 4)

    if not text:
        return jsonify({"error": "请输入新闻正文。"}), 400

    summary = textrank_summarize(text, ratio=ratio, max_sentences=max_sentences)
    return jsonify({"summary": summary})


@app.post("/api/qwen")
def api_qwen():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    model_path = payload.get("model_path") or "models/Qwen2.5-7B-Instruct"
    adapter_path = payload.get("adapter_path") or "."
    max_new_tokens = int(payload.get("max_new_tokens") or 220)
    temperature = float(payload.get("temperature") or 0.2)
    top_p = float(payload.get("top_p") or 0.9)
    repetition_penalty = float(payload.get("repetition_penalty") or 1.05)

    if not text:
        return jsonify({"error": "请输入新闻正文。"}), 400

    try:
        from scripts.qwen_lora_infer import infer_summary

        summary = infer_summary(
            text=text,
            model_path=model_path,
            adapter_path=adapter_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({"summary": summary})


@app.post("/api/rouge")
def api_rouge():
    payload = request.get_json(silent=True) or {}
    reference = (payload.get("reference") or "").strip()
    hypothesis = (payload.get("hypothesis") or "").strip()

    if not reference or not hypothesis:
        return jsonify({"error": "请同时提供参考摘要和待评估摘要。"}), 400

    return jsonify(score_summary(reference, hypothesis))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=True)
