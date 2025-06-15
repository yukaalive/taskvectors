import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import os
from datasets import Dataset


# 1. モデルとトークナイザーの準備
# model_id = "cyberagent/DeepSeek-R1-Distill-Qwen-14B-Japanese"

# # 量子化設定 (メモリ削減のため)
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True,
# )

# print(f"Loading model: {model_id}...")
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     # quantization_config=bnb_config,
#     device_map="cuda", # 自動で適切なデバイスにマップ
#     trust_remote_code=True, # カスタムコードの実行を許可
# ).to("cuda")
# # print("Model loaded.")

# print(f"Loading tokenizer: {model_id}...")
# # tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token # パディングトークンをEOSトークンに設定 (必要に応じて変更)
# tokenizer.padding_side = "right" # パディングの方向を右に設定
# # print("Tokenizer loaded.")

# # 2. PEFT (LoRA) 設定
# # モデルをLoRAに適応させるための準備
# model = prepare_model_for_kbit_training(model)

# lora_config = LoraConfig(
#     r=8,  # LoRAのランク (通常8, 16, 32, 64などを設定)
#     lora_alpha=16, # LoRAのスケーリング係数 (rの2倍程度が一般的)
#     target_modules=[  # LoRAを適用するモジュール (モデルによって異なる)
#         "q_proj",
#         "k_proj",
#         "v_proj",
#         "o_proj",
#         "gate_proj",
#         "up_proj",
#         "down_proj",
#     ],
#     bias="none", # バイアス層にLoRAを適用しない
#     lora_dropout=0.05, # ドロップアウト率
#     task_type="CAUSAL_LM", # タスクタイプは因果言語モデリング
# )

# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters() # 学習可能なパラメータ数を確認

# 3. データセットの準備と前処理
# Japanese-English Subtitle Corpusのパスを指定
# 仮に、データが以下のようなCSVファイル形式であると仮定します。
# 'japanese_text', 'english_text' の2つのカラムを持つCSVファイル
# 例: `data/japanese_english_subtitles.csv`
data_file = "translation_data.csv" # 実際のパスに置き換えてください

# CSVファイルを読み込む例
train_dataset = load_dataset("csv", data_files=data_file)
# # デモンストレーションのため、trainスプリットのみ使用
# train_dataset = train_dataset["train"]

# for i in range(len(train_dataset["train"])):

#     print(f"Row {i}: {train_dataset['train'][i]}")


print(train_dataset)