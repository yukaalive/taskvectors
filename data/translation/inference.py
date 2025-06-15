
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import os
from datasets import Dataset
output_dir = "./deepseek_qwen_fine_tuned"
os.makedirs(output_dir, exist_ok=True)
output_lora_dir = os.path.join(output_dir, "/home/yukaalive/2025workspace/task_vectors/4_icl_task_vectors/data/translation/deepseek_qwen_fine_tuned/checkpoint-30")


# 学習済みモデルのロードと推論例
print("\n--- Inference Example ---")
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,  # 推論時も同じ量子化設定を使用
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

from peft import PeftModel
fine_tuned_model = PeftModel.from_pretrained(base_model, output_lora_dir)
fine_tuned_model.eval()

inference_tokenizer = AutoTokenizer.from_pretrained(output_lora_dir, trust_remote_code=True)
inference_tokenizer.pad_token = inference_tokenizer.eos_token
inference_tokenizer.padding_side = "right"

def generate_translation(text, model, tokenizer):
    prompt = f"以下の日本語を英語に翻訳してください：{text}\n英語："
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    prompt_length = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
        )
    
    decoded_output = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    
    if "英語：" in decoded_output:
        return decoded_output.split("英語：", 1)[1].strip()
    return decoded_output.strip()

test_japanese_text = "この映画は本当に感動的でした。"
print(f"Japanese Input: {test_japanese_text}")
generated_english = generate_translation(test_japanese_text, fine_tuned_model, inference_tokenizer)
print(f"Generated English: {generated_english}")

test_japanese_text_2 = "どうすればあなたを助けることができますか？"
print(f"Japanese Input: {test_japanese_text_2}")
generated_english_2 = generate_translation(test_japanese_text_2, fine_tuned_model, inference_tokenizer)
print(f"Generated English: {generated_english_2}")
