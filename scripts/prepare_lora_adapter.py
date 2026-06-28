from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def write_adapter_config(adapter_dir: Path, base_model: str, lora_alpha: int, lora_dropout: float) -> Path:
    config = {
        "base_model_name_or_path": base_model,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": 16,
        "revision": None,
        "target_modules": DEFAULT_TARGET_MODULES,
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }
    output = adapter_dir / "adapter_config.json"
    output.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="为当前 LoRA safetensors 生成 PEFT adapter_config.json")
    parser.add_argument("--adapter-dir", default=".", help="包含 adapter_model.safetensors 的目录")
    parser.add_argument("--base-model", default="models/Qwen2.5-7B-Instruct")
    parser.add_argument("--lora-alpha", type=int, default=32, help="训练时的 lora_alpha；不确定时保留 32")
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    adapter_dir = Path(args.adapter_dir)
    if not (adapter_dir / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"未找到 {(adapter_dir / 'adapter_model.safetensors').resolve()}")

    output = write_adapter_config(adapter_dir, args.base_model, args.lora_alpha, args.lora_dropout)
    print(f"已生成 {output.resolve()}")


if __name__ == "__main__":
    main()
