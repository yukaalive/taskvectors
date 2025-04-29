import matplotlib.pyplot as plt
import numpy as np

# データの準備 - 各モデルのレイヤー別平均精度を計算
# LLaMA 7B (全14タスクの平均)
llama_7b_layers = list(range(32))  # 0-31のレイヤー
llama_7b_task_accuracies = np.array([
    # タスク1: translation_fr_en
    [0.14, 0.16, 0.10, 0.12, 0.18, 0.12, 0.14, 0.10, 0.10, 0.12, 0.08, 0.14, 0.54, 0.52, 0.60, 0.62, 
     0.56, 0.28, 0.08, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
     
    # タスク2: linguistic_present_simple_gerund
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.50, 0.74, 0.90, 0.90,
     0.42, 0.06, 0.04, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
     
    # タスク3: algorithmic_next_letter
    [0.12, 0.20, 0.18, 0.14, 0.26, 0.18, 0.18, 0.14, 0.36, 0.24, 0.38, 0.72, 0.94, 0.96, 0.98, 0.84,
     0.62, 0.36, 0.20, 0.10, 0.10, 0.10, 0.08, 0.06, 0.06, 0.06, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
     
    # タスク4: translation_es_en
    [0.14, 0.16, 0.12, 0.12, 0.14, 0.10, 0.20, 0.20, 0.20, 0.20, 0.12, 0.22, 0.74, 0.64, 0.78, 0.78,
     0.68, 0.28, 0.16, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.02],
     
    # タスク5: translation_en_fr
    [0.02, 0.18, 0.14, 0.18, 0.16, 0.18, 0.20, 0.24, 0.24, 0.24, 0.24, 0.26, 0.64, 0.56, 0.76, 0.74,
     0.50, 0.12, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.06, 0.04, 0.06, 0.02, 0.04],
     
    # タスク6: translation_en_es
    [0.10, 0.10, 0.08, 0.12, 0.12, 0.14, 0.16, 0.16, 0.14, 0.14, 0.14, 0.16, 0.68, 0.68, 0.84, 0.82,
     0.60, 0.10, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
     
    # タスク7: linguistic_present_simple_past_simple
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.36, 0.70, 0.92, 0.92,
     0.56, 0.06, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
     
    # タスク8: linguistic_plural_singular
    [0.16, 0.26, 0.16, 0.18, 0.22, 0.32, 0.34, 0.38, 0.40, 0.40, 0.40, 0.54, 0.74, 0.74, 0.80, 0.82,
     0.50, 0.14, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
     
    # タスク9: linguistic_antonyms
    [0.18, 0.10, 0.12, 0.04, 0.04, 0.02, 0.10, 0.26, 0.32, 0.60, 0.62, 0.78, 0.84, 0.84, 0.86, 0.80,
     0.50, 0.24, 0.16, 0.06, 0.02, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
     
    # タスク10: algorithmic_prev_letter
    [0.02, 0.00, 0.00, 0.02, 0.06, 0.02, 0.06, 0.02, 0.04, 0.08, 0.06, 0.14, 0.24, 0.28, 0.26, 0.20,
     0.32, 0.20, 0.10, 0.08, 0.08, 0.08, 0.08, 0.06, 0.06, 0.06, 0.08, 0.04, 0.06, 0.06, 0.08, 0.04],
     
    # タスク11: algorithmic_list_first
    [0.92, 0.88, 0.80, 0.88, 0.76, 0.80, 0.86, 0.84, 0.86, 0.80, 0.76, 0.98, 0.98, 0.96, 0.90, 0.94,
     0.48, 0.56, 0.42, 0.18, 0.14, 0.12, 0.10, 0.10, 0.08, 0.06, 0.06, 0.06, 0.06, 0.04, 0.06, 0.02],
     
    # タスク12: algorithmic_list_last
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.00, 0.00, 0.08, 0.08, 0.06, 0.62, 0.82, 0.86, 0.98, 0.90,
     0.78, 0.36, 0.20, 0.08, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.02, 0.02],
     
    # タスク13: algorithmic_to_upper
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.38, 0.96, 0.88, 0.54, 0.54,
     0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
     
    # タスク14: algorithmic_to_lower
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.70, 1.00, 0.98, 0.68, 0.52,
     0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
])

# MiniLLM 7B (全14タスクの平均)
minillm_7b_layers = list(range(32))
minillm_7b_task_accuracies = np.array([
    # タスク1: translation_fr_en
    [0.14, 0.14, 0.18, 0.14, 0.14, 0.20, 0.24, 0.30, 0.32, 0.28, 0.64, 0.58, 0.62, 0.64, 0.62, 0.56,
     0.42, 0.20, 0.06, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
     
    # タスク2: linguistic_present_simple_gerund
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.68, 0.88, 0.80, 0.60,
     0.46, 0.06, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
     
    # タスク3: algorithmic_next_letter
    [0.02, 0.00, 0.00, 0.00, 0.02, 0.14, 0.12, 0.04, 0.18, 0.36, 0.94, 0.96, 1.00, 1.00, 1.00, 0.96,
     0.96, 0.68, 0.42, 0.20, 0.24, 0.30, 0.24, 0.26, 0.02, 0.02, 0.02, 0.02, 0.04, 0.02, 0.02, 0.02],
     
    # タスク4: translation_es_en
    [0.10, 0.14, 0.12, 0.12, 0.16, 0.18, 0.22, 0.40, 0.46, 0.38, 0.78, 0.76, 0.76, 0.76, 0.80, 0.72,
     0.52, 0.20, 0.04, 0.00, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.00, 0.00, 0.02],
     
    # タスク5: translation_en_fr
    [0.14, 0.20, 0.20, 0.18, 0.16, 0.18, 0.20, 0.26, 0.22, 0.24, 0.26, 0.22, 0.20, 0.72, 0.70, 0.70,
     0.44, 0.06, 0.02, 0.02, 0.00, 0.02, 0.02, 0.02, 0.02, 0.02, 0.00, 0.00, 0.02, 0.04, 0.04, 0.04],
     
    # タスク6: translation_en_es
    [0.10, 0.12, 0.10, 0.10, 0.10, 0.10, 0.10, 0.14, 0.14, 0.14, 0.14, 0.12, 0.22, 0.72, 0.72, 0.60,
     0.36, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
     
    # タスク7: linguistic_present_simple_past_simple
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.00, 0.84, 0.96, 0.94, 0.70,
     0.56, 0.26, 0.06, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
     
    # タスク8: linguistic_plural_singular
    [0.20, 0.20, 0.24, 0.24, 0.28, 0.28, 0.26, 0.40, 0.38, 0.46, 0.42, 0.36, 0.60, 0.64, 0.74, 0.56,
     0.54, 0.20, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
     
    # タスク9: linguistic_antonyms
    [0.28, 0.20, 0.18, 0.20, 0.22, 0.20, 0.14, 0.40, 0.46, 0.64, 0.84, 0.84, 0.86, 0.88, 0.86, 0.76,
     0.54, 0.06, 0.06, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
     
    # タスク10: algorithmic_prev_letter
    [0.04, 0.06, 0.08, 0.04, 0.08, 0.08, 0.04, 0.00, 0.06, 0.04, 0.12, 0.16, 0.18, 0.24, 0.22, 0.28,
     0.26, 0.10, 0.06, 0.06, 0.04, 0.04, 0.04, 0.04, 0.06, 0.06, 0.06, 0.06, 0.06, 0.04, 0.06, 0.06],
     
    # タスク11: algorithmic_list_first
    [0.98, 0.94, 0.96, 0.90, 0.94, 0.92, 0.94, 0.98, 0.98, 0.96, 0.96, 0.96, 0.98, 0.96, 0.86, 0.56,
     0.44, 0.38, 0.14, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.08, 0.08, 0.10, 0.04, 0.06, 0.06, 0.04],
     
    # タスク12: algorithmic_list_last
    [0.00, 0.00, 0.00, 0.04, 0.00, 0.00, 0.00, 0.04, 0.04, 0.16, 0.86, 0.78, 0.82, 0.90, 0.90, 0.68,
     0.58, 0.38, 0.06, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
     
    # タスク13: algorithmic_to_upper
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.50, 0.46, 0.40, 0.00,
     0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
     
    # タスク14: algorithmic_to_lower
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.48, 0.08, 0.12, 0.02,
     0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
])

# LLaMA 13B (全14タスクの平均)
llama_13b_layers = list(range(40))
llama_13b_task_accuracies = np.array([
    # タスク1: translation_fr_en
    [0.04, 0.06, 0.06, 0.08, 0.08, 0.10, 0.08, 0.08, 0.26, 0.26, 0.32, 0.28, 0.50, 0.56, 0.58, 0.56, 0.62, 0.62, 0.58, 0.10,
     0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
     
    # タスク2: linguistic_present_simple_gerund
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.56, 0.64, 0.70, 0.60, 0.64, 0.42, 0.02,
     0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
     
    # タスク3: algorithmic_next_letter
    [0.14, 0.10, 0.16, 0.30, 0.28, 0.48, 0.34, 0.52, 0.58, 0.60, 0.58, 0.62, 0.86, 0.88, 0.92, 0.70, 0.62, 0.30, 0.20, 0.18,
     0.06, 0.08, 0.08, 0.04, 0.06, 0.04, 0.04, 0.06, 0.06, 0.08, 0.08, 0.08, 0.08, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.02],
     
    # タスク4: translation_es_en
    [0.04, 0.08, 0.04, 0.06, 0.14, 0.20, 0.18, 0.16, 0.50, 0.46, 0.50, 0.40, 0.68, 0.76, 0.78, 0.80, 0.74, 0.72, 0.54, 0.08,
     0.06, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.00, 0.02],
     
    # タスク5: translation_en_fr
    [0.10, 0.12, 0.12, 0.16, 0.18, 0.20, 0.20, 0.20, 0.20, 0.20, 0.22, 0.22, 0.24, 0.28, 0.30, 0.64, 0.60, 0.54, 0.30, 0.06,
     0.04, 0.04, 0.04, 0.04, 0.02, 0.02, 0.06, 0.06, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.06, 0.04, 0.04],
     
    # タスク6: translation_en_es
    [0.04, 0.06, 0.04, 0.06, 0.10, 0.12, 0.12, 0.12, 0.18, 0.18, 0.18, 0.16, 0.16, 0.20, 0.30, 0.72, 0.64, 0.64, 0.26, 0.00,
     0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
     
    # タスク7: linguistic_present_simple_past_simple
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.10, 0.06, 0.00, 0.00, 0.00, 0.00, 0.14, 0.88, 0.88, 0.96, 0.96, 0.94, 0.66, 0.16,
     0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
     
    # タスク8: linguistic_plural_singular
    [0.22, 0.20, 0.20, 0.24, 0.38, 0.40, 0.40, 0.40, 0.38, 0.40, 0.44, 0.40, 0.68, 0.86, 0.88, 0.84, 0.84, 0.82, 0.70, 0.50,
     0.20, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
     
    # タスク9: linguistic_antonyms
    [0.28, 0.32, 0.34, 0.32, 0.48, 0.42, 0.46, 0.50, 0.60, 0.56, 0.60, 0.66, 0.80, 0.86, 0.86, 0.84, 0.82, 0.64, 0.26, 0.06,
     0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.02],
     
    # タスク10: algorithmic_prev_letter
    [0.06, 0.10, 0.10, 0.10, 0.06, 0.04, 0.04, 0.06, 0.08, 0.08, 0.14, 0.12, 0.26, 0.28, 0.32, 0.40, 0.40, 0.32, 0.28, 0.24,
     0.14, 0.12, 0.10, 0.06, 0.06, 0.04, 0.04, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.02, 0.06, 0.04, 0.04, 0.04, 0.04, 0.02],
     
    # タスク11: algorithmic_list_first
    [0.88, 0.84, 0.88, 0.90, 0.98, 1.00, 0.98, 0.88, 0.94, 0.90, 0.86, 0.84, 0.84, 0.98, 0.76, 0.52, 0.58, 0.52, 0.46, 0.50,
     0.54, 0.14, 0.16, 0.12, 0.10, 0.12, 0.14, 0.16, 0.10, 0.12, 0.12, 0.12, 0.14, 0.12, 0.12, 0.10, 0.12, 0.10, 0.06, 0.02],
     
    # タスク12: algorithmic_list_last
    [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.12, 0.18, 0.30, 0.42, 0.92, 0.98, 0.98, 0.92, 0.86, 0.76, 0.76, 0.78,
     0.32, 0.06, 0.04, 0.02, 0.00, 0.04, 0.04, 0.02, 0.00, 0.00, 0.00, 0.02, 0.02, 0.04, 0.02, 0.00, 0.04, 0.00, 0.02, 0.02],
     
    # タスク13: algorithmic_to_upper
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.12, 0.14, 0.04, 0.00, 0.00, 0.00, 0.00, 0.00, 0.96, 0.90, 0.10, 0.02, 0.04, 0.00, 0.00,
     0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
     # タスク14: algorithmic_to_lower
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.06, 0.14, 0.04, 0.00, 0.02, 0.10, 0.02, 0.66, 1.00, 0.94, 0.30, 0.06, 0.02, 0.02, 0.00,
     0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
])

# 各モデルの平均精度を計算
llama_7b_avg = np.mean(llama_7b_task_accuracies, axis=0)
minillm_7b_avg = np.mean(minillm_7b_task_accuracies, axis=0)
llama_13b_avg = np.mean(llama_13b_task_accuracies, axis=0)

# 相対レイヤー位置を計算（0-1の範囲に正規化）
llama_7b_relative_layers = np.array(llama_7b_layers) / (len(llama_7b_layers) - 1)
minillm_7b_relative_layers = np.array(minillm_7b_layers) / (len(minillm_7b_layers) - 1)
llama_13b_relative_layers = np.array(llama_13b_layers) / (len(llama_13b_layers) - 1)

# モデル間比較グラフを作成
plt.figure(figsize=(12, 8))

# 絶対レイヤー位置でのプロット
plt.subplot(1, 2, 1)
plt.plot(llama_7b_layers, llama_7b_avg, label='LLaMA 7B', color='blue', linewidth=2)
plt.plot(minillm_7b_layers, minillm_7b_avg, label='MiniLLM 7B', color='green', linewidth=2)
#plt.plot(llama_13b_layers[:32], llama_13b_avg[:32], label='LLaMA 13B (first 32 layers)', color='red', linewidth=2, linestyle='--')
plt.plot(llama_13b_layers, llama_13b_avg, label='LLaMA 13B', color='red', linewidth=2)

# ピーク精度のマーキング
peak_llama_7b = llama_7b_layers[np.argmax(llama_7b_avg)]
peak_minillm_7b = minillm_7b_layers[np.argmax(minillm_7b_avg)]
peak_llama_13b = llama_13b_layers[np.argmax(llama_13b_avg)]

plt.axvline(x=peak_llama_7b, color='blue', linestyle=':', alpha=0.7)
plt.axvline(x=peak_minillm_7b, color='green', linestyle=':', alpha=0.7)
plt.axvline(x=peak_llama_13b, color='red', linestyle=':', alpha=0.7)

plt.annotate(f'Peak: Layer {peak_llama_7b}', 
             xy=(peak_llama_7b, llama_7b_avg[np.argmax(llama_7b_avg)]),
             xytext=(peak_llama_7b + 1, llama_7b_avg[np.argmax(llama_7b_avg)] - 0.05),
             color='blue', fontsize=10)
             
plt.annotate(f'Peak: Layer {peak_minillm_7b}', 
             xy=(peak_minillm_7b, minillm_7b_avg[np.argmax(minillm_7b_avg)]),
             xytext=(peak_minillm_7b + 1, minillm_7b_avg[np.argmax(minillm_7b_avg)] - 0.05),
             color='green', fontsize=10)
             
plt.annotate(f'Peak: Layer {peak_llama_13b}', 
             xy=(peak_llama_13b, llama_13b_avg[np.argmax(llama_13b_avg)]),
             xytext=(peak_llama_13b - 8, llama_13b_avg[np.argmax(llama_13b_avg)] - 0.05),
             color='red', fontsize=10)

plt.title('Average Task Accuracy by Layer', fontsize=16)
plt.xlabel('Layer', fontsize=14)
plt.ylabel('Average Accuracy', fontsize=14)
plt.ylim(0, 1.0)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# 相対レイヤー位置でのプロット
plt.subplot(1, 2, 2)
plt.plot(llama_7b_relative_layers, llama_7b_avg, label='LLaMA 7B', color='blue', linewidth=2)
plt.plot(minillm_7b_relative_layers, minillm_7b_avg, label='MiniLLM 7B', color='green', linewidth=2)
plt.plot(llama_13b_relative_layers, llama_13b_avg, label='LLaMA 13B', color='red', linewidth=2)

# 相対位置でのピーク精度のマーキング
rel_peak_llama_7b = np.argmax(llama_7b_avg) / (len(llama_7b_avg) - 1)
rel_peak_minillm_7b = np.argmax(minillm_7b_avg) / (len(minillm_7b_avg) - 1)
rel_peak_llama_13b = np.argmax(llama_13b_avg) / (len(llama_13b_avg) - 1)

plt.axvline(x=rel_peak_llama_7b, color='blue', linestyle=':', alpha=0.7)
plt.axvline(x=rel_peak_minillm_7b, color='green', linestyle=':', alpha=0.7)
plt.axvline(x=rel_peak_llama_13b, color='red', linestyle=':', alpha=0.7)

plt.annotate(f'Peak: {rel_peak_llama_7b:.2f}', 
             xy=(rel_peak_llama_7b, llama_7b_avg[np.argmax(llama_7b_avg)]),
             xytext=(rel_peak_llama_7b + 0.05, llama_7b_avg[np.argmax(llama_7b_avg)] - 0.05),
             color='blue', fontsize=10)
             
plt.annotate(f'Peak: {rel_peak_minillm_7b:.2f}', 
             xy=(rel_peak_minillm_7b, minillm_7b_avg[np.argmax(minillm_7b_avg)]),
             xytext=(rel_peak_minillm_7b + 0.05, minillm_7b_avg[np.argmax(minillm_7b_avg)] - 0.05),
             color='green', fontsize=10)
             
plt.annotate(f'Peak: {rel_peak_llama_13b:.2f}', 
             xy=(rel_peak_llama_13b, llama_13b_avg[np.argmax(llama_13b_avg)]),
             xytext=(rel_peak_llama_13b - 0.15, llama_13b_avg[np.argmax(llama_13b_avg)] - 0.05),
             color='red', fontsize=10)

plt.title('Average Task Accuracy by Layer (Relative Position)', fontsize=16)
plt.xlabel('Relative Layer Position (0-1)', fontsize=14)
plt.ylabel('Average Accuracy', fontsize=14)
plt.ylim(0, 1.0)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# 全体のタイトル
plt.suptitle('Comparison of Layer-wise Average Accuracy across Models', fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('model_average_accuracy_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# モデルごとのピーク精度とその位置を統計情報として表示
print(f"LLaMA 7B - Peak accuracy: {np.max(llama_7b_avg):.4f} at layer {np.argmax(llama_7b_avg)} (relative position: {rel_peak_llama_7b:.4f})")
print(f"MiniLLM 7B - Peak accuracy: {np.max(minillm_7b_avg):.4f} at layer {np.argmax(minillm_7b_avg)} (relative position: {rel_peak_minillm_7b:.4f})")
print(f"LLaMA 13B - Peak accuracy: {np.max(llama_13b_avg):.4f} at layer {np.argmax(llama_13b_avg)} (relative position: {rel_peak_llama_13b:.4f})")

# さらに詳細な分析：レイヤーを前半・中盤・後半に分けた場合の平均精度を計算
def calculate_region_accuracy(accuracy_array):
    n = len(accuracy_array)
    early = np.mean(accuracy_array[:n//3])
    middle = np.mean(accuracy_array[n//3:2*n//3])
    late = np.mean(accuracy_array[2*n//3:])
    return early, middle, late

llama_7b_early, llama_7b_middle, llama_7b_late = calculate_region_accuracy(llama_7b_avg)
minillm_7b_early, minillm_7b_middle, minillm_7b_late = calculate_region_accuracy(minillm_7b_avg)
llama_13b_early, llama_13b_middle, llama_13b_late = calculate_region_accuracy(llama_13b_avg)

print("\nAverage accuracy by model region:")
print(f"LLaMA 7B - Early: {llama_7b_early:.4f}, Middle: {llama_7b_middle:.4f}, Late: {llama_7b_late:.4f}")
print(f"MiniLLM 7B - Early: {minillm_7b_early:.4f}, Middle: {minillm_7b_middle:.4f}, Late: {minillm_7b_late:.4f}")
print(f"LLaMA 13B - Early: {llama_13b_early:.4f}, Middle: {llama_13b_middle:.4f}, Late: {llama_13b_late:.4f}")