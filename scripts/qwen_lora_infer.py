from __future__ import annotations

import argparse
from pathlib import Path


PROMPT_TEMPLATE = """你是中文新闻文本自动摘要系统。请从下面新闻正文中生成简洁、准确、覆盖核心事实的中文摘要。

要求：
1. 保留人物、机构、时间、地点、事件和结果等关键信息。
2. 不编造原文没有的信息。
3. 输出 2 到 5 句中文摘要。

新闻正文：
{text}
"""


def infer_summary(
    text: str,
    model_path: str = "models/Qwen2.5-7B-Instruct",
    adapter_path: str = ".",
    max_new_tokens: int = 220,
    temperature: float = 0.2,
    top_p: float = 0.9,
    repetition_penalty: float = 1.05,
) -> str:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model_dir = Path(model_path)
    adapter_dir = Path(adapter_path)

    if not model_dir.exists():
        raise FileNotFoundError(f"未找到基础模型目录：{model_dir.resolve()}")
    if not (adapter_dir / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"未找到 LoRA 权重：{(adapter_dir / 'adapter_model.safetensors').resolve()}")
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(
            "未找到 adapter_config.json。请使用官方 LoRA 配置文件，或运行 "
            "`python scripts/prepare_lora_adapter.py --base-model models/Qwen2.5-7B-Instruct` 生成基础配置。"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    messages = [{"role": "user", "content": PROMPT_TEMPLATE.format(text=text.strip())}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen2.5-7B-Instruct + LoRA 中文摘要推理")
    parser.add_argument("--text", "-t", required=True, help="新闻正文")
    parser.add_argument("--model-path", default="models/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter-path", default=".")
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    args = parser.parse_args()

    print(
        infer_summary(
            text=args.text,
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
    )


if __name__ == "__main__":
    main()
