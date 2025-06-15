import torch
from transformers import AutoConfig
from core.models.llm_loading import MODEL_PATHS

def check_hidden_dimensions():
    """各モデルのhidden layersの次元数を確認する"""
    
    # 現在の実験設定で使用されているモデル
    models_to_check = [
        ("youko", "8B"),
        # 他のモデルも必要に応じて追加可能
        # ("Qwen", "14B"),
        # ("llama", "7B"),
    ]
    
    print("=== Hidden Layers Dimensions ===")
    
    for model_type, model_variant in models_to_check:
        try:
            model_path = MODEL_PATHS[model_type][model_variant]
            print(f"\nModel: {model_type} {model_variant}")
            print(f"Path: {model_path}")
            
            # モデルの設定を読み込み
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            
            # hidden_sizeを取得
            if hasattr(config, 'hidden_size'):
                hidden_size = config.hidden_size
                print(f"Hidden size: {hidden_size}")
            elif hasattr(config, 'd_model'):
                hidden_size = config.d_model
                print(f"Hidden size (d_model): {hidden_size}")
            else:
                print("Hidden size not found in config")
                continue
                
            # レイヤー数も確認
            if hasattr(config, 'num_hidden_layers'):
                num_layers = config.num_hidden_layers
                print(f"Number of layers: {num_layers}")
            elif hasattr(config, 'n_layer'):
                num_layers = config.n_layer
                print(f"Number of layers (n_layer): {num_layers}")
            else:
                print("Number of layers not found in config")
                
            # その他の関連する設定も表示
            if hasattr(config, 'intermediate_size'):
                print(f"Intermediate size (FFN): {config.intermediate_size}")
            if hasattr(config, 'num_attention_heads'):
                print(f"Number of attention heads: {config.num_attention_heads}")
                
        except Exception as e:
            print(f"Error loading {model_type} {model_variant}: {e}")
    
    print("\n=== Task Vector Context ===")
    print("タスクベクトルでは、各レイヤーのhidden stateの最後のトークン位置の値を使用します。")
    print("形状: (num_datasets, num_layers, hidden_size)")
    print("- num_datasets: テストデータセットの数")
    print("- num_layers: モデルのレイヤー数（embedding層を除く）")
    print("- hidden_size: 各レイヤーの隠れ状態の次元数")

if __name__ == "__main__":
    check_hidden_dimensions()
