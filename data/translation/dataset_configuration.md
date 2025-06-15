# 翻訳タスク用データセット設定ガイド

## 概要


#### データセットサイズ
- **ファイル名**: `translation_data.csv`
- **ファイルサイズ**: 215.17 MB
- **行数**: 2,801,389行（約280万行）


#### 1. データサイズ削減

**作成したスクリプト**: `prepare_data.py`

```python
def prepare_translation_data(input_file="translation_data.csv", 
                           output_file="translation_data_small.csv",
                           num_samples=10000):
    # 280万行から1万行に削減
```

**削減果**:
- **出力ファイル**: `translation_data_small.csv`
- **ファイルサイズ**: 0.77 MB
- **行数**: 10,000行


## データセット仕様

### ファイル構造

```csv
input,output
"you are back, aren't you, harold?","あなたは戻ったのね ハロルド?"
"my opponent is shark.","俺の相手は シャークだ。"
```

### 学習用データ変換

```python
# プロンプト形式
prompt = f"以下の日本語を英語に翻訳してください：{japanese_text}\n英語："
target = english_text

# Qwen Chat形式
formatted_text = f"<|im_start|>user\n{prompt}{target}<|im_end|>"
```

## ファイル一覧

### データファイル
- `translation_data.csv` - 元の大容量データセット（280万行）
- `translation_data_small.csv` - 削減済みデータセット（1万行）

### スクリプトファイル
- `prepare_data.py` - データサイズ削減スクリプト
- `sft20250614_fixed.py` - ファインチューニング
## ステップ

### 1. データ準備

```bash
cd /path/to/translation/
python prepare_data.py
```

### 2. ファインチューニング実行

```bash
python sft20250614_fixed.py
```

## コンフィグ

### LoRA設定
```python
lora_config = LoraConfig(
    r=8,                    # ランク
    lora_alpha=16,          # Alpha値
    lora_dropout=0.05,      # ドロップアウト率
    task_type="CAUSAL_LM",  # タスクタイプ
)
```

### トレーニング設定
```python
training_args = TrainingArguments(
    num_train_epochs=3,              # エポック数
    per_device_train_batch_size=16,  # バッチサイズ
    gradient_accumulation_steps=2,   # 勾配累積
    learning_rate=2e-4,              # 学習率
    max_length=512,                  # 最大シーケンス長
)
```

