# Qwen2.5-7B LCSTS LoRA Adapter

This directory contains the final LoRA adapter trained for Chinese news summarization on a filtered 5k LCSTS subset.

## Contents

- `adapter_model.safetensors`: LoRA adapter weights, stored with Git LFS.
- `adapter_config.json`: PEFT adapter configuration. The base model is `Qwen/Qwen2.5-7B-Instruct`.
- `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`: tokenizer and chat template copied from the training output.
- `training_args.yaml`, `train_results.json`, `all_results.json`: training configuration and final metrics.
- `trainer_state.json`, `trainer_log.jsonl`, `training_loss.png`: training state and logs for reference.

## Training Summary

- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Fine-tuning type: LoRA
- Stage: SFT
- Epochs: 2.0
- Global steps: 1334
- Train loss: 2.0462
- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- Quantization during training: 4-bit bitsandbytes

## Load Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_model = "Qwen/Qwen2.5-7B-Instruct"
adapter_path = "adapters/qwen2.5-7b-lcsts-lora"

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    quantization_config=quantization_config,
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()
```

The base model weights are not included in this repository.
