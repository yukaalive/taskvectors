import matplotlib.pyplot as plt
import numpy as np

# データの準備（各モデルのレイヤー別精度）
# LLaMA 7B
llama_7b_layers = list(range(32))  # 0-31のレイヤー
llama_7b_accuracy = [
    # translation_fr_en タスクのレイヤー別精度
    [0.14, 0.16, 0.10, 0.12, 0.18, 0.12, 0.14, 0.10, 0.10, 0.12, 0.08, 0.14, 0.54, 0.52, 0.60, 0.62, 
     0.56, 0.28, 0.08, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    
    # linguistic_present_simple_gerund タスクのレイヤー別精度
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.50, 0.74, 0.90, 0.90,
     0.42, 0.06, 0.04, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    
    # algorithmic_next_letter タスクのレイヤー別精度
    [0.12, 0.20, 0.18, 0.14, 0.26, 0.18, 0.18, 0.14, 0.36, 0.24, 0.38, 0.72, 0.94, 0.96, 0.98, 0.84,
     0.62, 0.36, 0.20, 0.10, 0.10, 0.10, 0.08, 0.06, 0.06, 0.06, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04]
]

# MiniLLM 7B
minillm_7b_layers = list(range(32))  # 0-31のレイヤー
minillm_7b_accuracy = [
    # translation_fr_en タスクのレイヤー別精度
    [0.14, 0.14, 0.18, 0.14, 0.14, 0.20, 0.24, 0.30, 0.32, 0.28, 0.64, 0.58, 0.62, 0.64, 0.62, 0.56,
     0.42, 0.20, 0.06, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    
    # linguistic_present_simple_gerund タスクのレイヤー別精度
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.68, 0.88, 0.80, 0.60,
     0.46, 0.06, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    
    # algorithmic_next_letter タスクのレイヤー別精度
    [0.02, 0.00, 0.00, 0.00, 0.02, 0.14, 0.12, 0.04, 0.18, 0.36, 0.94, 0.96, 1.00, 1.00, 1.00, 0.96,
     0.96, 0.68, 0.42, 0.20, 0.24, 0.30, 0.24, 0.26, 0.02, 0.02, 0.02, 0.02, 0.04, 0.02, 0.02, 0.02]
]

# LLaMA 13B
llama_13b_layers = list(range(40))  # 0-39のレイヤー
llama_13b_accuracy = [
    # translation_fr_en タスクのレイヤー別精度
    [0.04, 0.06, 0.06, 0.08, 0.08, 0.10, 0.08, 0.08, 0.26, 0.26, 0.32, 0.28, 0.50, 0.56, 0.58, 0.56, 0.62, 0.62, 0.58, 0.10,
     0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
    
    # linguistic_present_simple_gerund タスクのレイヤー別精度
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.56, 0.64, 0.70, 0.60, 0.64, 0.42, 0.02,
     0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    
    # algorithmic_next_letter タスクのレイヤー別精度
    [0.14, 0.10, 0.16, 0.30, 0.28, 0.48, 0.34, 0.52, 0.58, 0.60, 0.58, 0.62, 0.86, 0.88, 0.92, 0.70, 0.62, 0.30, 0.20, 0.18,
     0.06, 0.08, 0.08, 0.04, 0.06, 0.04, 0.04, 0.06, 0.06, 0.08, 0.08, 0.08, 0.08, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.02]
]

# モデルごとに平均精度を計算
llama_7b_avg = np.mean(llama_7b_accuracy, axis=0)
minillm_7b_avg = np.mean(minillm_7b_accuracy, axis=0)
llama_13b_avg = np.mean(llama_13b_accuracy, axis=0)

# 相対レイヤー位置を計算（0-1の範囲に正規化）
llama_7b_relative_layers = np.array(llama_7b_layers) / (len(llama_7b_layers) - 1)
minillm_7b_relative_layers = np.array(minillm_7b_layers) / (len(minillm_7b_layers) - 1)
llama_13b_relative_layers = np.array(llama_13b_layers) / (len(llama_13b_layers) - 1)

# 4つのサブプロットを作成（3つのモデル個別 + 1つの平均比較）
fig, axs = plt.subplots(2, 2, figsize=(18, 14))

# タイトルと軸ラベルのフォントサイズを設定
title_fontsize = 16
label_fontsize = 14
legend_fontsize = 12

# LLaMA 7B
axs[0, 0].plot(llama_7b_layers, llama_7b_accuracy[0], label='Translation Task', marker='o', markersize=4)
axs[0, 0].plot(llama_7b_layers, llama_7b_accuracy[1], label='Linguistic Task', marker='s', markersize=4)
axs[0, 0].plot(llama_7b_layers, llama_7b_accuracy[2], label='Algorithmic Task', marker='^', markersize=4)
axs[0, 0].plot(llama_7b_layers, llama_7b_avg, label='Average Accuracy', linestyle='--', color='black', linewidth=2)
axs[0, 0].set_title('LLaMA 7B Model', fontsize=title_fontsize)
axs[0, 0].set_xlabel('Layer Number', fontsize=label_fontsize)
axs[0, 0].set_ylabel('Accuracy', fontsize=label_fontsize)
axs[0, 0].set_ylim(0, 1.0)
axs[0, 0].legend(loc='upper right', fontsize=legend_fontsize)
axs[0, 0].grid(True, alpha=0.3)

# MiniLLM 7B
axs[0, 1].plot(minillm_7b_layers, minillm_7b_accuracy[0], label='Translation Task', marker='o', markersize=4)
axs[0, 1].plot(minillm_7b_layers, minillm_7b_accuracy[1], label='Linguistic Task', marker='s', markersize=4)
axs[0, 1].plot(minillm_7b_layers, minillm_7b_accuracy[2], label='Algorithmic Task', marker='^', markersize=4)
axs[0, 1].plot(minillm_7b_layers, minillm_7b_avg, label='Average Accuracy', linestyle='--', color='black', linewidth=2)
axs[0, 1].set_title('MiniLLM 7B Model', fontsize=title_fontsize)
axs[0, 1].set_xlabel('Layer Number', fontsize=label_fontsize)
axs[0, 1].set_ylabel('Accuracy', fontsize=label_fontsize)
axs[0, 1].set_ylim(0, 1.0)
axs[0, 1].legend(loc='upper right', fontsize=legend_fontsize)
axs[0, 1].grid(True, alpha=0.3)

# LLaMA 13B
axs[1, 0].plot(llama_13b_layers, llama_13b_accuracy[0], label='Translation Task', marker='o', markersize=4)
axs[1, 0].plot(llama_13b_layers, llama_13b_accuracy[1], label='Linguistic Task', marker='s', markersize=4)
axs[1, 0].plot(llama_13b_layers, llama_13b_accuracy[2], label='Algorithmic Task', marker='^', markersize=4)
axs[1, 0].plot(llama_13b_layers, llama_13b_avg, label='Average Accuracy', linestyle='--', color='black', linewidth=2)
axs[1, 0].set_title('LLaMA 13B Model', fontsize=title_fontsize)
axs[1, 0].set_xlabel('Layer Number', fontsize=label_fontsize)
axs[1, 0].set_ylabel('Accuracy', fontsize=label_fontsize)
axs[1, 0].set_ylim(0, 1.0)
axs[1, 0].legend(loc='upper right', fontsize=legend_fontsize)
axs[1, 0].grid(True, alpha=0.3)

# モデル間の平均精度比較（相対レイヤー位置を使用）
axs[1, 1].plot(llama_7b_relative_layers, llama_7b_avg, label='LLaMA 7B', color='blue', linewidth=2)
axs[1, 1].plot(minillm_7b_relative_layers, minillm_7b_avg, label='MiniLLM 7B', color='green', linewidth=2)
axs[1, 1].plot(llama_13b_relative_layers, llama_13b_avg, label='LLaMA 13B', color='red', linewidth=2)
axs[1, 1].set_title('Model Comparison (Average Accuracy)', fontsize=title_fontsize)
axs[1, 1].set_xlabel('Relative Layer Position (0-1)', fontsize=label_fontsize)
axs[1, 1].set_ylabel('Average Accuracy', fontsize=label_fontsize)
axs[1, 1].set_ylim(0, 1.0)
axs[1, 1].legend(loc='upper right', fontsize=legend_fontsize)
axs[1, 1].grid(True, alpha=0.3)

# ピーク精度の位置に垂直線を追加
peak_llama_7b = np.argmax(llama_7b_avg) / (len(llama_7b_avg) - 1)
peak_minillm_7b = np.argmax(minillm_7b_avg) / (len(minillm_7b_avg) - 1)
peak_llama_13b = np.argmax(llama_13b_avg) / (len(llama_13b_avg) - 1)

axs[1, 1].axvline(x=peak_llama_7b, color='blue', linestyle=':', alpha=0.7)
axs[1, 1].axvline(x=peak_minillm_7b, color='green', linestyle=':', alpha=0.7)
axs[1, 1].axvline(x=peak_llama_13b, color='red', linestyle=':', alpha=0.7)

# 全体のタイトル
fig.suptitle('Layer-wise Task Accuracy Comparison across Language Models', fontsize=20)

plt.tight_layout(rect=[0, 0, 1, 0.97])  # suptitleのスペースを確保
plt.savefig('model_comparison_with_averages.png', dpi=300, bbox_inches='tight')
plt.show()