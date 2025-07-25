import math
import os
from typing import Literal, Tuple
from peft import PeftModel

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from accelerate import init_empty_weights
from accelerate.utils.modeling import infer_auto_device_map, get_balanced_memory

from core.models.utils.llm_layers import get_layers, get_layers_path

BASE_KWARGS = {
    "torch_dtype": torch.float32,
    "trust_remote_code": True,
}

GPU_KWARGS = {
    **BASE_KWARGS,
    # "load_in_8bit": True,
    # "device_map": "auto",
}



LlamaVariant = Literal["huggingface"]
LlamaSize = Literal["7B","13B"]


def _setup_tokenizer(tokenizer: PreTrainedTokenizer) -> None:
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token


def llama_local_path(variant: LlamaVariant, size: LlamaSize) -> str:
    llama_dir = os.environ["LLAMA_DIR"]
    return f"{llama_dir}/{variant}/{size}"


def get_local_path(model_type: str, model_variant: str) -> str:
    if model_type == "llama":
        return llama_local_path("huggingface", model_variant)

    model_path = get_model_path(model_type, model_variant)
    username, model_name = model_path.split("/")

    huggingface_cache_dir = os.environ["TRANSFORMERS_CACHE"]
    return f"{huggingface_cache_dir}/models--{username}--{model_name}"


def get_model_path(model_type: str, model_variant: str) -> str:
    model_path = MODEL_PATHS[model_type][model_variant]
    return model_path


def get_latest_checkpoint_path(base_path: str) -> str:
    """最新のチェックポイントディレクトリを取得"""
    if not os.path.exists(base_path):
        raise ValueError(f"Base path does not exist: {base_path}")
    
    checkpoint_dirs = [d for d in os.listdir(base_path) if d.startswith('checkpoint-') and os.path.isdir(os.path.join(base_path, d))]
    if not checkpoint_dirs:
        raise ValueError(f"No checkpoint directories found in {base_path}")
    
    # 数字部分でソートして最新を取得
    latest = max(checkpoint_dirs, key=lambda x: int(x.split('-')[1]))
    return os.path.join(base_path, latest)


def _get_falcon_device_map() -> dict[str, int]:
    num_devices = torch.cuda.device_count()
    device_map = {
        "transformer.word_embeddings": 0,
        "lm_head": 0,
        "transformer.ln_f": 0,
    }
    num_layers = 60
#    num_layers_per_device = math.ceil(num_layers / (num_devices - 1))
    num_layers_per_device = 1
    device_map.update({f"transformer.h.{i}": (i // num_layers_per_device + 1) for i in range(num_layers)})
    return device_map


def _create_device_map(model_path: str) -> dict[str, int]:
    with init_empty_weights():
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    layer_class = get_layers(model)[0].__class__.__name__

    max_memory = get_balanced_memory(model, no_split_module_classes=[layer_class])
    max_memory[0] = 0
    base_device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=[layer_class])

    num_devices = torch.cuda.device_count()

    layers_path = get_layers_path(model)

    # device_map_lm_head = {k: v for k, v in base_device_map.items() if "lm_head" in k}
    # device_map_emb = {k: v for k, v in base_device_map.items() if "emb" in k}
    device_map_layers = {k: v for k, v in base_device_map.items() if k.startswith(layers_path)}
    device_map_other = {k: v for k, v in base_device_map.items() if k not in device_map_layers}

    # place the other layers on device 0
    device_map_other = {k: 0 for k in device_map_other}
    # split the layers evenly across the other devices (1-num_devices)
    num_layers = len(device_map_layers)
    #num_layers_per_device = math.ceil(num_layers / (num_devices - 1))
    num_layers_per_device = 1
    device_map_layers = {k: (i // num_layers_per_device + 1) for i, k in enumerate(device_map_layers)}

    device_map = {**device_map_other, **device_map_layers}

    return device_map


def load_model(model_type: str, model_variant: str):
    # model_path = "meta-llama/Llama-2-7b-hf"
    model_path = MODEL_PATHS[model_type][model_variant]
    model_paty = get_model_path(model_type, model_variant)

    kwargs = GPU_KWARGS.copy()  # 元のディクショナリをコピー
    
    # ファインチューニングしたモデルの場合
    if model_type == "Qwen" and model_variant == "14B_j_sft":
        # ベースモデルのパス
        base_model_path = "cyberagent/DeepSeek-R1-Distill-Qwen-14B-Japanese"
        kwargs["device_map"] = _create_device_map(base_model_path)
        
        print(f"Loading base model: {base_model_path}")
        model = AutoModelForCausalLM.from_pretrained(base_model_path, **kwargs)
        
        # LoRAアダプターのパスを取得
        ft_base_path = "/home/yukaalive/2025workspace/task_vectors/4_icl_task_vectors/data/translation/deepseek_qwen_fine_tuned"
        try:
            lora_adapter_path = get_latest_checkpoint_path(ft_base_path)
            print(f"Loading LoRA adapter from: {lora_adapter_path}")
            
            # LoRAアダプターを適用
            model = PeftModel.from_pretrained(model, lora_adapter_path)
            model = model.merge_and_unload()  # LoRAアダプターをマージして通常のモデルに変換
            print("LoRA adapter successfully merged")
            
        except Exception as e:
            print(f"Warning: Failed to load LoRA adapter: {e}")
            print("Using base model without fine-tuning")
    else:
        # nekomataモデルの場合、trust_remote_code=Trueを追加（すでにBASE_KWARGSに含まれている）
        kwargs["device_map"] = _create_device_map(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    
    model = model.eval()  # check if this is necessary

    return model


def load_tokenizer(model_type: str, model_variant: str) -> PreTrainedTokenizer:
    # model_path = "meta-llama/Llama-2-7b-hf"
    model_path = MODEL_PATHS[model_type][model_variant]
    model_paty = get_model_path(model_type, model_variant)
    #tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    
    # ファインチューニングしたモデルの場合
    if model_type == "Qwen" and model_variant == "14B_j_sft":
        # ベースモデルのトークナイザーを使用
        base_model_path = "cyberagent/DeepSeek-R1-Distill-Qwen-14B-Japanese"
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, padding_side="left", trust_remote_code=True)
    # nekomataモデルの場合だけtrust_remote_code=Trueを追加
    elif model_type == "nekomata":
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", trust_remote_code=True)
    # Qwenモデルの場合もtrust_remote_code=Trueを追加
    elif model_type == "Qwen":
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        
    _setup_tokenizer(tokenizer)

    return tokenizer


def load_model_and_tokenizer(
    model_type: str, model_variant: str
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    tokenizer = load_tokenizer(model_type, model_variant)
    model = load_model(model_type, model_variant)

    return model, tokenizer

MODEL_PATHS = {
    "llama": {
        "7B": "meta-llama/Llama-2-7b-hf",
        "13B": "meta-llama/Llama-2-13b-hf"
    },
    "minillm": {
        "7B": "MiniLLM/MiniLLM-Llama-7B"
    },
    "Qwen":{
        "14B_j": "cyberagent/DeepSeek-R1-Distill-Qwen-14B-Japanese",
        "14B" : "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "14B_j_sft": "/home/yukaalive/2025workspace/task_vectors/4_icl_task_vectors/data/translation/deepseek_qwen_fine_tuned"
        
    },
    "youko":{
        "8B": "rinna/llama-3-youko-8b"
    },  
    "nekomata":{
        "7B": "rinna/nekomata-7b"
    },
    "llm-jp":{
        "13B":"llm-jp/llm-jp-13b-v2.0"
    },
    "swallow":{
        "7B":"tokyotech-llm/Swallow-7b-hf"
    },
    "xalma":{
        "13B":"haoranxu/X-ALMA-13B-Group6"
    },
    "gpt-j":{
        "6B":"EleutherAI/gpt-j-6b"
    },
    "pythia":{
        "2.8B":"EleutherAI/pythia-2.8b",
        "12B":"EleutherAI/pythia-12b"
    }
}
