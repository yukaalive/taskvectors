# llm-jp/llm-jp-3-13bを4bit量子化のqLoRA設定でロードする

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from unsloth import FastLanguageModel
import torch
max_seq_length = 512 # コンテクスト長。unslothでRoPEをサポートしているので可変。
dtype = None # Noneにしておけば自動で設定される。
load_in_4bit = True # 4bit量子化を使うかどうか。メモリ使用量を大幅に削減できるが精度が若干低下することがある

model_id = "cyberagent/DeepSeek-R1-Distill-Qwen-14B-Japanese"
new_model_id = "cyberagent-deepseek-r1-distill-qwen-14b-sft-1.0" # 新しいモデルの名前
# FastLanguageModelでHugging Faceからモデルをロードし、インスタンスを作成
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_id,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    trust_remote_code=True,
)

# SFT用のモデルを用意
model = FastLanguageModel.get_peft_model(
    model,
    r = 32, # LoRAランク。小さい値ほどパラメータ数が減りメモリ効率が上がるが、精度も低下する可能性がある
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32, # 低ランク行列のスケーリング係数。値が大きいほど、追加されるパラメータの影響が大きくなる
    lora_dropout = 0.05, # LoRAの低ランク行列に適用するドロップアウト率
    bias = "none", # バイアスの処理方法。
    use_gradient_checkpointing = "unsloth", # Unslothによる勾配計算のメモリ効率化
    random_state = 3407, # 乱数のシード値
    use_rslora = False, # rslora（別のLoRA手法）を使うかどうか。ここでは使用しない。
    loftq_config = None, # LoRAと量子化の組み合わせ設定。ここでは使用しない。
    max_seq_length = max_seq_length, # 最大シーケンス長。
)