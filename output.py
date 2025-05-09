import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# データの読み込み（タブ区切りテキストとしてコピーしたデータをファイルに保存した場合）
# CSVファイルがない場合は、以下のデータ部分をコピーして使用

# データを直接定義
data = [
    ["translation_fr_en", 0.08, 0.14, 0.10, 0.08, 0.10, 0.08, 0.10, 0.16, 0.12, 0.14, 0.12, 0.24, 0.48, 0.50, 0.52, 0.52, 0.54, 0.26, 0.16, 0.08, 0.06, 0.08, 0.06, 0.06, 0.06, 0.04, 0.04, 0.06, 0.04, 0.04, 0.04, 0.00],
    ["linguistic_present_simple_gerund", 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.44, 0.70, 0.90, 0.90, 0.56, 0.04, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    ["knowledge_country_capital", 0.14, 0.12, 0.14, 0.16, 0.14, 0.16, 0.16, 0.18, 0.24, 0.28, 0.24, 0.40, 0.34, 0.36, 0.70, 0.62, 0.62, 0.48, 0.34, 0.04, 0.00, 0.00, 0.04, 0.04, 0.04, 0.04, 0.04, 0.06, 0.06, 0.06, 0.06, 0.00],
    ["algorithmic_next_letter", 0.04, 0.20, 0.20, 0.20, 0.38, 0.20, 0.22, 0.12, 0.52, 0.28, 0.46, 0.70, 0.92, 0.88, 0.94, 0.92, 0.60, 0.36, 0.24, 0.16, 0.08, 0.06, 0.02, 0.02, 0.02, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    ["translation_es_en", 0.18, 0.24, 0.20, 0.16, 0.22, 0.24, 0.22, 0.24, 0.14, 0.18, 0.10, 0.26, 0.58, 0.58, 0.60, 0.62, 0.64, 0.34, 0.26, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.02, 0.02, 0.00, 0.00, 0.00],
    ["translation_en_fr", 0.12, 0.12, 0.14, 0.16, 0.16, 0.18, 0.26, 0.26, 0.26, 0.28, 0.26, 0.28, 0.66, 0.74, 0.72, 0.76, 0.58, 0.08, 0.04, 0.00, 0.00, 0.04, 0.04, 0.02, 0.02, 0.02, 0.04, 0.02, 0.02, 0.04, 0.02, 0.00],
    ["translation_en_es", 0.04, 0.02, 0.04, 0.06, 0.06, 0.06, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.66, 0.70, 0.84, 0.82, 0.52, 0.04, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    ["linguistic_present_simple_past_simple", 0.02, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.06, 0.58, 0.76, 0.92, 0.92, 0.42, 0.12, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    ["linguistic_plural_singular", 0.20, 0.18, 0.18, 0.20, 0.20, 0.24, 0.24, 0.24, 0.26, 0.26, 0.26, 0.46, 0.68, 0.62, 0.76, 0.76, 0.54, 0.16, 0.06, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    ["linguistic_antonyms", 0.32, 0.28, 0.32, 0.10, 0.14, 0.06, 0.24, 0.26, 0.38, 0.64, 0.62, 0.66, 0.74, 0.80, 0.80, 0.78, 0.60, 0.38, 0.22, 0.06, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    ["knowledge_person_language", 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.06, 0.06, 0.02, 0.68, 0.76, 0.70, 0.60, 0.54, 0.22, 0.12, 0.12, 0.12, 0.12, 0.14, 0.12, 0.16, 0.16, 0.18, 0.20, 0.20, 0.10],
    ["knowledge_location_continent", 0.04, 0.02, 0.02, 0.04, 0.04, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.10, 0.10, 0.06, 0.22, 0.24, 0.62, 0.60, 0.68, 0.60, 0.62, 0.60, 0.58, 0.64, 0.68, 0.66, 0.68, 0.68, 0.70, 0.68, 0.64, 0.32],
    ["knowledge_location_religion", 0.16, 0.12, 0.14, 0.18, 0.14, 0.16, 0.14, 0.14, 0.24, 0.24, 0.24, 0.26, 0.24, 0.20, 0.48, 0.62, 0.74, 0.70, 0.70, 0.66, 0.62, 0.60, 0.52, 0.64, 0.62, 0.56, 0.56, 0.58, 0.60, 0.50, 0.52, 0.10],
    ["algorithmic_prev_letter", 0.02, 0.00, 0.00, 0.04, 0.04, 0.00, 0.02, 0.00, 0.02, 0.02, 0.02, 0.08, 0.10, 0.12, 0.10, 0.14, 0.14, 0.12, 0.06, 0.06, 0.04, 0.02, 0.02, 0.06, 0.04, 0.06, 0.08, 0.06, 0.04, 0.06, 0.06, 0.08],
    ["algorithmic_list_first", 0.92, 0.88, 0.80, 0.88, 0.76, 0.80, 0.86, 0.84, 0.86, 0.80, 0.76, 0.98, 0.98, 0.96, 0.90, 0.94, 0.48, 0.56, 0.42, 0.18, 0.14, 0.12, 0.10, 0.10, 0.08, 0.06, 0.06, 0.06, 0.06, 0.04, 0.06, 0.00],
    ["algorithmic_list_last", 0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.00, 0.00, 0.08, 0.08, 0.06, 0.62, 0.82, 0.86, 0.98, 0.92, 0.78, 0.34, 0.20, 0.08, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.02, 0.00],
    ["algorithmic_to_upper", 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.24, 0.88, 0.86, 0.48, 0.50, 0.06, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    ["algorithmic_to_lower", 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.00, 0.06, 0.58, 1.00, 0.98, 0.82, 0.68, 0.16, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
]

# データフレームに変換
columns = ['task'] + [f'layer_{i+1}' for i in range(32)]
df = pd.DataFrame(data, columns=columns)

# タスク名を行インデックスに設定
df_analysis = df.copy()
df_analysis.set_index('task', inplace=True)

# 1. 各タスクの最適レイヤーを特定
best_layers = []
for task in df_analysis.index:
    max_acc = df_analysis.loc[task].max()
    best_layer_indices = df_analysis.loc[task][df_analysis.loc[task] == max_acc].index.tolist()
    best_layers.append({
        'task': task,
        'best_accuracy': max_acc,
        'best_layers': [int(layer.split('_')[1]) for layer in best_layer_indices]
    })

# 2. タスクの種類ごとに分類
task_categories = {
    'linguistic': [task for task in df_analysis.index if 'linguistic' in task],
    'knowledge': [task for task in df_analysis.index if 'knowledge' in task],
    'algorithmic': [task for task in df_analysis.index if 'algorithmic' in task],
    'translation': [task for task in df_analysis.index if 'translation' in task]
}

# 3. ヒートマップでデータを視覚化
plt.figure(figsize=(20, 10))
sns.heatmap(df_analysis, cmap='viridis', annot=True, fmt='.2f', linewidths=0.5)
plt.title('Task Accuracy by Layer')
plt.xlabel('Layer')
plt.ylabel('Task')
plt.tight_layout()
plt.savefig('task_layer_heatmap.png')

# 4. カテゴリごとの最適レイヤーの分布
category_best_layers = {}
for category, tasks in task_categories.items():
    category_best_layers[category] = []
    for task_info in best_layers:
        if task_info['task'] in tasks:
            category_best_layers[category].extend(task_info['best_layers'])

plt.figure(figsize=(15, 8))
for i, (category, layers) in enumerate(category_best_layers.items(), 1):
    plt.subplot(2, 2, i)
    plt.hist(layers, bins=range(1, 33), alpha=0.7)
    plt.title(f'Best Layers for {category.capitalize()} Tasks')
    plt.xlabel('Layer')
    plt.ylabel('Frequency')
    plt.xticks(range(1, 33, 2))
    plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('category_best_layers.png')

# 5. 各レイヤーの平均精度をプロット
avg_by_layer = df_analysis.mean()

plt.figure(figsize=(15, 6))
plt.plot(range(1, 33), avg_by_layer, marker='o', linewidth=2)
plt.fill_between(range(1, 33), avg_by_layer - avg_by_layer.std(), avg_by_layer + avg_by_layer.std(), alpha=0.2)
plt.grid(True, alpha=0.3)
plt.title('Average Accuracy by Layer Across All Tasks')
plt.xlabel('Layer')
plt.ylabel('Average Accuracy')
plt.xticks(range(1, 33, 1))
plt.savefig('avg_accuracy_by_layer.png')

# 6. タスクごとの最適レイヤーと精度をテーブルとして表示
best_layers_df = pd.DataFrame(best_layers)
best_layers_df['best_layers'] = best_layers_df['best_layers'].apply(lambda x: ', '.join(map(str, x)))
print("Best Layers for Each Task:")
print(best_layers_df.sort_values('best_accuracy', ascending=False))

# 7. 各カテゴリの最適レイヤーの統計
print("\nStatistics of Best Layers by Category:")
for category, layers in category_best_layers.items():
    if layers:
        print(f"{category.capitalize()}:")
        print(f"  Mean: {np.mean(layers):.2f}")
        print(f"  Median: {np.median(layers):.1f}")
        print(f"  Min: {min(layers)}")
        print(f"  Max: {max(layers)}")
        print(f"  Most common: {pd.Series(layers).value_counts().index[0]}")
        print()

# 8. レイヤーの性質によるクラスタリング
# 各レイヤーのパフォーマンスの特性を理解するための正規化データを作成
normalized_df = df_analysis.apply(zscore, axis=1)

# レイヤーごとの特性を可視化
plt.figure(figsize=(15, 8))
sns.clustermap(normalized_df.T, cmap='vlag', figsize=(15, 10),
               row_cluster=True, col_cluster=True, 
               yticklabels=True, xticklabels=True)
plt.savefig('layer_clustering.png')
