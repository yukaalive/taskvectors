import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import os
from datasets import Dataset


# 1. モデルとトークナイザーの準備
model_id = "cyberagent/DeepSeek-R1-Distill-Qwen-14B-Japanese"

# 量子化設定 (メモリ削減のため)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print(f"Loading model: {model_id}...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,  # 量子化設定を有効化
    device_map="auto",  # "cuda"から"auto"に変更
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,  # データ型を明示的に指定
)
print("Model loaded.")

print(f"Loading tokenizer: {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("Tokenizer loaded.")

# 2. PEFT (LoRA) 設定
# 量子化されたモデルをLoRAに適応させるための準備
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 3. データセットの準備と前処理
data_file = "translation_data.csv"

train_dataset = load_dataset("csv", data_files=data_file)
train_dataset = train_dataset["train"].rename_column("input", "japanese_text").rename_column("output", "english_text")

def preprocess_function(examples):
    inputs = [f"以下の日本語を英語に翻訳してください：{jp_text}\n英語：" for jp_text in examples["japanese_text"]]
    targets = [en_text for en_text in examples["english_text"]]

    def format_qwen_chat(prompt, target):
        return f"<|im_start|>user\n{prompt}{target}<|im_end|>"
    
    formatted_texts = [format_qwen_chat(p, t) for p, t in zip(inputs, targets)]

    # return_tensors="pt"を削除し、辞書形式で返す
    tokenized_inputs = tokenizer(
        formatted_texts,
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    
    # labelsをinput_idsのコピーとして設定
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    
    return tokenized_inputs

print("Preprocessing dataset...")
tokenized_train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names  # 元のカラムを削除
)
print("Dataset preprocessed.")

# 4. トレーニングの設定と実行
output_dir = "./deepseek_qwen_fine_tuned"
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=16,  # バッチサイズを小さく
    gradient_accumulation_steps=2,  # 勾配累積を増やして実質バッチサイズを維持
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=True,
    logging_steps=10,
    save_steps=10,
    save_total_limit=2,
    overwrite_output_dir=True,
    report_to="none",
    gradient_checkpointing=True,
    dataloader_pin_memory=False,  # メモリ使用量を削減
    remove_unused_columns=False,  # 不要なカラムの削除を無効化
)

# カスタムTrainerクラスを定義（オプション）
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        カスタム損失計算関数
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        else:
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    tokenizer=tokenizer,
)

# print("Starting training...")
# trainer.train()
# print("Training finished.")
output_lora_dir = os.path.join(output_dir, "/home/yukaalive/2025workspace/task_vectors/4_icl_task_vectors/data/translation/deepseek_qwen_fine_tuned/checkpoint-20")
# # 5. ファインチューニングモデルの保存
# output_lora_dir = os.path.join(output_dir, "lora_adapters")
# os.makedirs(output_lora_dir, exist_ok=True)
# model.save_pretrained(output_lora_dir)
# tokenizer.save_pretrained(output_lora_dir)
# print(f"Fine-tuned LoRA adapters and tokenizer saved to {output_lora_dir}")

# 学習済みモデルのロードと推論例
print("\n--- Inference Example ---")
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    #quantization_config=bnb_config,  # 推論時も同じ量子化設定を使用
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

# def generate_translation(text, model, tokenizer):
#     prompt = f"以下の日本語を英語に翻訳してください：{text}\n英語："
#     formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>"
    
#     inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
#     prompt_length = inputs["input_ids"].shape[1]
    
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=50,
#             num_beams=1,
#             do_sample=False,
#             pad_token_id=tokenizer.eos_token_id,
#             no_repeat_ngram_size=3,
#         )
    
#     decoded_output = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    
#     if "英語：" in decoded_output:
#         return decoded_output.split("英語：", 1)[1].strip()
#     return decoded_output.strip()

# test_japanese_text = "この映画は本当に感動的でした。"
# print(f"Japanese Input: {test_japanese_text}")
# generated_english = generate_translation(test_japanese_text, fine_tuned_model, inference_tokenizer)
# print(f"Generated English: {generated_english}")

# test_japanese_text_2 = "どうすればあなたを助けることができますか？"
# print(f"Japanese Input: {test_japanese_text_2}")
# generated_english_2 = generate_translation(test_japanese_text_2, fine_tuned_model, inference_tokenizer)
# print(f"Generated English: {generated_english_2}")
def generate_translation(text, model, tokenizer):
    # 学習時と同じプロンプト形式を使用
    prompt = f"以下の日本語を英語に翻訳してください：{text}\n英語："
    
    # 学習時は <|im_start|>user\n{prompt}{target}<|im_end|> の形式
    # 推論時は <|im_start|>user\n{prompt} で終わらせ、続きを生成させる
    formatted_prompt = f"<|im_start|>user\n{prompt}"
    
    print(f"使用するプロンプト: '{formatted_prompt}'")
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    prompt_length = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            min_new_tokens=3,           # 最低3トークンは生成
            do_sample=True,             # サンプリングを有効
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),  # 学習時の終了トークン
            repetition_penalty=1.1,
        )
    
    print(f"生成後のトークン数: {outputs.shape[1]}")
    
    # 全体の出力を確認
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(f"全体の出力: {full_output}")
    
    # 生成された部分のみ
    generated_part = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    print(f"生成された部分: '{generated_part}'")
    
    # <|im_end|> を除去
    if "<|im_end|>" in generated_part:
        generated_part = generated_part.split("<|im_end|>")[0]
    
    return generated_part.strip()

test_japanese_text = "この映画は本当に感動的でした。"
print(f"Japanese Input: {test_japanese_text}")
generated_english = generate_translation(test_japanese_text, fine_tuned_model, inference_tokenizer)
print(f"Generated English: {generated_english}")

test_japanese_text_2 = "どうすればあなたを助けることができますか？"
print(f"Japanese Input: {test_japanese_text_2}")
generated_english_2 = generate_translation(test_japanese_text_2, fine_tuned_model, inference_tokenizer)
print(f"Generated English: {generated_english_2}")