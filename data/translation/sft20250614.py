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
    # quantization_config=bnb_config,
    device_map="cuda", # 自動で適切なデバイスにマップ
    trust_remote_code=True, # カスタムコードの実行を許可
).to("cuda")
print("Model loaded.")

print(f"Loading tokenizer: {model_id}...")
# tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token # パディングトークンをEOSトークンに設定 (必要に応じて変更)
tokenizer.padding_side = "right" # パディングの方向を右に設定
print("Tokenizer loaded.")

# 2. PEFT (LoRA) 設定
# モデルをLoRAに適応させるための準備
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,  # LoRAのランク (通常8, 16, 32, 64などを設定)
    lora_alpha=16, # LoRAのスケーリング係数 (rの2倍程度が一般的)
    target_modules=[  # LoRAを適用するモジュール (モデルによって異なる)
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    bias="none", # バイアス層にLoRAを適用しない
    lora_dropout=0.05, # ドロップアウト率
    task_type="CAUSAL_LM", # タスクタイプは因果言語モデリング
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters() # 学習可能なパラメータ数を確認

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

train_dataset = train_dataset["train"].rename_column("input", "japanese_text").rename_column("output", "english_text")

# データの前処理関数
# モデルが理解しやすい形式に変換
def preprocess_function(examples):
    # 日本語と英語のテキストを結合し、モデルへの入力形式に合わせる
    # 翻訳タスクの場合、例えば "日本語: <日本語文> 英語: <英語文>" のような形式が考えられます
    # 今回は、対話形式でモデルが日本語を英語に翻訳するように促すプロンプトを構築します
    # 例: 「以下の日本語を英語に翻訳してください：[日本語テキスト]\n英語：[英語テキスト]」
    
    # 応答生成としてファインチューニングするため、inputとlabelを同じにする
    inputs = [f"以下の日本語を英語に翻訳してください：{jp_text}\n英語：" for jp_text in examples["japanese_text"]]
    targets = [en_text for en_text in examples["english_text"]]

    # プロンプトとターゲットを結合して、モデルの学習データとする
    # Qwenモデルは、ChatML形式のような特定のプロンプト形式を使用することがあります。
    # ここでは一般的な形式を使用しますが、必要に応じて調整してください。
    
    # モデルの期待するプロンプト形式に変換する関数を定義
    def format_qwen_chat(prompt, target):
        # QwenのChatMLフォーマットの例（モデルのバージョンによって異なる場合があります）
        # <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{target}<|im_end|>
        return f"<|im_start|>user\n{prompt}{target}<|im_end|>"
    
    formatted_texts = [format_qwen_chat(p, t) for p, t in zip(inputs, targets)]

    # トークナイズ
    tokenized_inputs = tokenizer(
        formatted_texts,
        max_length=512, # 最大シーケンス長 (モデルの制約とデータに合わせて調整)
        truncation=True,
        padding="max_length", # バッチ内の最長シーケンスにパディング
        return_tensors="pt"
    )
    
    # ラベルは入力と同じ
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
    
    # パディングトークンを無視するためのマスクを設定 (オプション、通常はTrainerが処理)
    # ここでは、特に処理は行わない
    
    return tokenized_inputs

print("Preprocessing dataset...")
# map関数でデータセット全体に前処理を適用
tokenized_train_dataset = train_dataset.map(
    preprocess_function,
    batched=True # バッチで処理することで高速化
    # remove_columns=train_dataset.column_names # 元のカラムを削除
)
print("Dataset preprocessed.")

# 4. トレーニングの設定と実行
output_dir = "./deepseek_qwen_fine_tuned"
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3, # エポック数
    per_device_train_batch_size=2, # デバイスごとのバッチサイズ (GPUメモリに合わせて調整)
    gradient_accumulation_steps=4, # 勾配累積ステップ (実質バッチサイズを大きくする)
    learning_rate=2e-4, # 学習率
    weight_decay=0.01, # 重み減衰
    fp16=True, # 混合精度学習 (GPU使用時に推奨)
    logging_steps=10, # ログ出力の頻度
    save_steps=500, # チェックポイント保存の頻度
    save_total_limit=2, # 保存するチェックポイントの最大数
    overwrite_output_dir=True,
    report_to="none", # ログレポートを無効化 (wandbなどを使用する場合は設定)
    gradient_checkpointing=True, # 勾配チェックポインティング (メモリ削減)
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    tokenizer=tokenizer,
)

print("Starting training...")
trainer.train()
print("Training finished.")

# 5. ファインチューニングモデルの保存
# PEFTモデルは、ベースモデルとは別にLoRAアダプターのみを保存します
# 読み込み時は、ベースモデルを読み込んだ後にアダプターを適用します
output_lora_dir = os.path.join(output_dir, "lora_adapters")
os.makedirs(output_lora_dir, exist_ok=True)
model.save_pretrained(output_lora_dir)
tokenizer.save_pretrained(output_lora_dir)
print(f"Fine-tuned LoRA adapters and tokenizer saved to {output_lora_dir}")

# 学習済みモデルのロードと推論例
print("\n--- Inference Example ---")
# ベースモデルを再度ロード
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16, # 学習時と同じDTypeでロード
    trust_remote_code=True
)

# 保存したLoRAアダプターをロード
from peft import PeftModel
fine_tuned_model = PeftModel.from_pretrained(base_model, output_lora_dir)
fine_tuned_model = fine_tuned_model.merge_and_unload() # LoRAアダプターをマージして通常のモデルに変換 (オプション、推論時のみ)
fine_tuned_model.eval() # 推論モードに設定

inference_tokenizer = AutoTokenizer.from_pretrained(output_lora_dir, trust_remote_code=True)
inference_tokenizer.pad_token = inference_tokenizer.eos_token
inference_tokenizer.padding_side = "right"


# 推論関数
def generate_translation(text, model, tokenizer):
    prompt = f"以下の日本語を英語に翻訳してください：{text}\n英語："
    # Qwen ChatML形式に変換 (推論時も同じ形式を使用)
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # プロンプトの長さを取得し、生成の最大長を調整
    prompt_length = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50, # 生成する新しいトークンの最大数
            num_beams=1, # ビームサーチの数 (1でGreedy Search)
            do_sample=False, # サンプリングしない (確定的な結果)
            pad_token_id=tokenizer.eos_token_id, # パディングトークンIDを指定
            no_repeat_ngram_size=3, # n-gramの繰り返しを避ける
        )
    
    # 入力部分を除いた部分をデコード
    # Qwenの場合、<|im_start|>assistant\n...<|im_end|> の部分をデコードする必要がある
    decoded_output = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    
    # 応答が "英語：" で始まると仮定
    # ここでは、`format_qwen_chat`の形式からAssistantの応答部分のみを抽出するロジックが必要
    # 通常、モデルがAssistantの応答を生成し始める前に、<|im_start|>assistant\n が出力されるので、
    # その後の部分を抽出する
    
    # Qwenのデコードの際、もしAssistantの部分がそのまま返ってくる場合は、そこから抽出
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