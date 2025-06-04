import os
import pickle
from typing import List
import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from core.config import FIGURES_DIR

from scripts.figures.helpers import (
    MODEL_DISPLAY_NAME_MAPPING,
    load_main_results,
    extract_accuracies,
    create_accuracies_df,
    create_grouped_accuracies_df,
)


def filter_tasks_with_low_icl_accuracy(grouped_accuracies_df, regular_accuracy_threshold=0.20):
    mask = (grouped_accuracies_df["Regular"] - grouped_accuracies_df["Baseline"]) >= regular_accuracy_threshold
    filtered_task_accuracies_df = grouped_accuracies_df[mask].copy()

    # print the excluded model,task pairs, Hypothesis by commas
    if not mask.all():
        print(
            "Excluded:",
            grouped_accuracies_df[~mask][["model", "task_name", "Regular"]].apply(
                lambda x: f"({x['model']}, {x['task_name']}): {x['Regular']:.2f}", axis=1
            ),
        )
    print("Num excluded / total:", (~mask).sum(), "/", len(grouped_accuracies_df))

    return filtered_task_accuracies_df

def plot_accuracies_per_model_per_task(grouped_accuracies_df, task_names: List[str] = None, task_types: List[str] = None, output_dir: str = None):
    """
    各タスクごとに、各モデルの精度をバーグラフでプロットする関数
    
    Args:
        grouped_accuracies_df: グループ化された精度データフレーム
        task_names: プロットするタスク名のリスト（例: "ja_en", "en_ja"）
        task_types: プロットするタスクタイプのリスト（例: "translation"）
        output_dir: 出力ディレクトリ
    """
    
    if output_dir is None:
        output_dir = os.path.join(FIGURES_DIR, "per_task_bar_plots")
    
    os.makedirs(output_dir, exist_ok=True)
    
    filtered_task_accuracies_df = grouped_accuracies_df.copy()
    #columns_to_plot = ["Baseline","Hypothesis", "Regular"]
    columns_to_plot = ["Hypothesis", "Regular"]
    # タスク名とタスクタイプが両方指定されている場合、それらの組み合わせのみを処理
    if task_names is not None and task_types is not None:
        for task_type in task_types:
            for task_name in task_names:
                # このタスクタイプとタスク名の組み合わせのデータを抽出
                task_df = filtered_task_accuracies_df[
                    (filtered_task_accuracies_df["task_type"] == task_type) & 
                    (filtered_task_accuracies_df["task_name"] == task_name)
                ]
                
                
                # タスクのフルネームを作成（表示用）
                full_task_name = f"{task_type}_{task_name}"
                print(f"タスク '{full_task_name}' データのサイズ: {len(task_df)}")
                # if len(task_df) > 0:
                #     print(f"データフレーム内のモデル名: {sorted(task_df['model'].unique())}")
                #     print(f"最初の数行のサンプル:\n{task_df[['model', 'Hypothesis', 'Regular']].head()}")
                # else:
                #     print(f"警告: タスク '{full_task_name}' にデータがありません！")
                # モデルをソート
                model_names = [m for m in [
                    "Qwen 14B",
                    "Pythia 2.8B",
                    "Llama2 7B", 
                    "Pythia 12B",
                    "Llama2 13B",
                    "GPT-J 6B", 
                    "Qwen 14B_j",
                    "Swallow 7B", 
                    "Xalma 13B",
                    "Youko 8B"
                    ] if m in task_df["model"].unique()]
                num_models = len(model_names)
                
                # データフレームをモデルでインデックス化
                task_df_indexed = task_df.set_index("model")
                
                # プロットの設定
                plt.rcParams.update({"font.size": 14})
                fig, ax = plt.subplots(figsize=(8, 6))
                
                bar_width = 0.3
                hatches = ["/", "\\", "|"]
                
                # 各メソッド（Baseline, Hypothesis, Regular）ごとにバーをプロット
                for j, column in enumerate(columns_to_plot):
                    means = [task_df_indexed.loc[model, column] if model in task_df_indexed.index else np.nan for model in model_names]
                    y_positions = np.arange(len(means)) + (j - 1) * bar_width
                    #y_positions = np.arange(len(means)) + (j * 0.5)  # 元: (j - 0.5) * bar_width
                    
                    ax.barh(
                        y_positions,
                        means,
                        height=bar_width,
                        capsize=2,
                        color=["blue", "green"][j],
                        edgecolor="white",
                        hatch=hatches[j] * 2,
                    )
                
                # Y軸のラベルを設定
                ax.set_yticks(np.arange(num_models))
                ax.set_yticklabels([model_name for model_name in model_names])
                ax.set_yticklabels(ax.get_yticklabels(), rotation=45, ha="right")
                
                # X軸の設定
                ax.set_xlabel("Accuracy")
                ax.set_xlim([0.0, 1.0])
                
                # タイトルを設定
                # ax.set_title(f"{task_type.capitalize()}: {task_name.replace('_', ' ').title()}")
                
                # 凡例を追加
                legend_elements = [
                    # Patch(facecolor="grey", edgecolor="white", hatch=hatches[0] * 2, label="Baseline"),
                    Patch(facecolor="green", edgecolor="white", hatch=hatches[0] * 2, label="Regular ICL"),
                    Patch(facecolor="blue", edgecolor="white", hatch=hatches[1] * 2, label="Task Vectors"),
                ]
                ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
                
                # ファイル名で使用できる安全なタスク名を作成
                safe_task_name = full_task_name.replace("/", "_")
                
                # 図を保存
                save_path = os.path.join(output_dir, f"model_accuracies_{safe_task_name}.png")
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close()
                
                print(f"タスク '{full_task_name}' の棒グラフを保存しました: {save_path}")
    else:
        print("警告: task_namesとtask_typesの両方を指定してください。")

def plot_accuracy_by_layer_per_task_grouped(results, task_names: List[str] = None, normalize_x_axis: bool = False, output_dir: str = None):
    """
    各タスクごとに、モデルグループ別に各レイヤーでの精度をプロットする関数
    
    Args:
        results: 実験結果の辞書
        task_names: プロットするタスクの名前リスト（Noneの場合は最初のモデルのすべてのタスク）
        normalize_x_axis: レイヤー番号を正規化するかどうか
        output_dir: 出力ディレクトリ（Noneの場合はFIGURES_DIR）
    """
    if output_dir is None:
        output_dir = os.path.join(FIGURES_DIR, "per_task_grouped_plots")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # モデルグループの定義
    model_groups = [
        {
            "name": "group1",
            "models": ["youko_8B","xalma_13B", "swallow_7B","Qwen_j_14B"],
            "display_name": "japanese LLMs"
        },
        {
            "name": "group2",
            "models": ["gpt-j_6B","llama_13B","pythia_12B","llama_7B","pythia_2.8B","Qwen_14B"],
            "display_name": "non-japanese LLMs"
        },
        # {
        #     "name": "group3",
        #     "models": ["pythia_2.8B", "pythia_12B"],
        #     "display_name": "Pythia"
        # }
    ]
    
    # タスク名のデフォルト設定
    if task_names is None:
        # 利用可能な最初のモデルからタスク名を取得
        for group in model_groups:
            for model_name in group["models"]:
                if model_name in results:
                    task_names = list(results[model_name].keys())
                    break
            if task_names is not None:
                break
    
    # マーカーを定義
    markers = ["o", "^", "s", "P", "X", "D", "v", "*", "p", "h"]
    
    # レイヤー数の定義
    num_layers = {
        "llama_7B": 32,
        "llama_13B": 32,
        # 必要に応じて他のモデルも追加
    }
    
    # 各タスクごとにグラフを作成
    for task_name in task_names:
        # 各モデルグループごとにグラフを生成
        for group in model_groups:
            group_name = group["name"]
            group_display_name = group["display_name"]
            group_models = group["models"]
            
            # モデルグループのモデルが1つでもデータが存在するか確認
            has_data = False
            for model_name in group_models:
                if model_name in results and task_name in results[model_name]:
                    has_data = True
                    break
            
            if not has_data:
                continue
            
            # 新しい図を作成
            plt.figure(figsize=(10, 5))
            plt.rc("font", size=16)
            
            # 各モデルをプロット
            for idx, model_name in enumerate(group_models):
                if model_name not in results or task_name not in results[model_name]:
                    continue
                
                # レイヤー別の精度データを取得
                task_results = results[model_name][task_name]
                dev_accuracy_by_layer = task_results["tv_dev_accruacy_by_layer"]
                
                # レイヤー番号とその精度値を取得
                layers = np.array(list(dev_accuracy_by_layer.keys()))
                accuracies = np.array(list(dev_accuracy_by_layer.values()))
                
                # X軸の値（必要に応じて正規化）
                x_values = layers
                if normalize_x_axis and model_name in num_layers:
                    x_values = x_values / num_layers[model_name]
                
                # プロットを描画
                plt.plot(
                    x_values,
                    accuracies,
                    marker=markers[idx % len(markers)],  
                    markersize=10,
                    label=MODEL_DISPLAY_NAME_MAPPING.get(model_name, model_name),
                    alpha=0.8,
                )
            
            # グラフの装飾
            plt.xlabel("Layer")
            plt.ylabel("Accuracy")
            plt.ylim(0.0, 1.0)
            plt.legend(loc="upper right")
            
            # タスク名とグループ名でグラフのタイトルを設定
            #plt.title(f"{task_name.replace('_', ' ').title()} - {group_display_name}")
            
            # ファイル名で使用できる安全なタスク名を作成
            safe_task_name = task_name.replace("/", "_")
            
            # 図を保存
            save_path = os.path.join(output_dir, f"accuracy_per_layer_{safe_task_name}_{group_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

# つかってない
def plot_avg_accuracies_per_model(grouped_accuracies_df):
    filtered_task_accuracies_df = filter_tasks_with_low_icl_accuracy(grouped_accuracies_df)

    columns_to_plot = ["Baseline", "Hypothesis", "Regular"]

    # Calculate average accuracy and std deviation for each model
    df_agg = filtered_task_accuracies_df.groupby("model")[columns_to_plot].agg("mean")

    # Plotting

    # Sort the model names, firsts by the base name, then by the size (e.g. "Pythia 6.9B" < "Pythia 13B", "LLaMA 7B" < "LLaMA 13B")
    model_names = df_agg.index.unique()
    num_models = len(model_names)
    # model_names = sorted(model_names, key=lambda x: (x.split(" ")[0], float(x.split(" ")[1][:-1])))
    model_names = [m for m in ["gpt-j_6B", "pythia_2.8B", "pythia_12B", "llama_7B", "llama_13B", "swallow_7B", "xalma_13B", "youko_8B", "Qwen_14B", "Qwen_j_14B"] if m in df_agg.index.unique()]

    plt.rcParams.update({"font.size": 14})  # Set font size

    fig, ax = plt.subplots(figsize=(6, 6))

    bar_width = 0.3
    hatches = ["/", "\\", "|"]
    for j, column in enumerate(columns_to_plot):
        means = df_agg[column]
        y_positions = np.arange(len(means)) + (j - 1) * bar_width
        # make sure to show the model names from the index as the y ticks
        ax.barh(
            y_positions,
            means,
            height=bar_width,
            capsize=2,
            color=["grey", "blue", "green"][j],
            edgecolor="white",
            hatch=hatches[j] * 2,
        )

    # set the y ticks to be the model names, not numbers
    ax.set_yticks(np.arange(num_models))
    ax.set_yticklabels([model_name for model_name in model_names])
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45, ha="right")

    ax.set_xlabel("Accuracy")
    ax.set_xlim([0.0, 1.0])

    # show legend below the plot
    legend_elements = [
        Patch(facecolor="grey", edgecolor="white", hatch=hatches[0] * 2, label="Baseline"),
        Patch(facecolor="green", edgecolor="white", hatch=hatches[2] * 2, label="Regular"),
        Patch(facecolor="blue", edgecolor="white", hatch=hatches[1] * 2, label="Hypothesis"),
    ]
    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)

    # plt.tight_layout()

    # save the figure
    save_path = os.path.join(FIGURES_DIR, "main_experiment_results_per_model.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_accuracy_by_layer(results, model_names: List[str], normalize_x_axis: bool = False, filename_suffix: str = ""):
    plt.figure(figsize=(10, 5))

    plt.rc("font", size=16)

    # plt.title(f"Average Accuracy by Layer")

    regular_accuracy_threshold = 0.20

    num_layers = {
        "llama_7B": 32,
        "llama_7B": 32,
        "llama_13B": 32,
    }

    # Define different markers for each model
#    markers = ["o", "^", "s", "P", "X", "D", "v"]
    markers = ["o", "^", "s", "P", "X", "D", "v", "*", "p", "h"] 

    for idx, model_name in enumerate(model_names):
        min_num_layers = min(
            len(results[model_name][task_name]["tv_dev_accruacy_by_layer"]) for task_name in results[model_name]
        )
        all_tv_dev_accruacy_by_layer = np.array(
            [
                np.array(list(results[model_name][task_name]["tv_dev_accruacy_by_layer"].values())[:min_num_layers])
                for task_name in results[model_name]
            ]
        )

        all_tv_dev_accruacy_by_layer = all_tv_dev_accruacy_by_layer[
            all_tv_dev_accruacy_by_layer.max(axis=-1) > regular_accuracy_threshold
        ]

        mean_tv_dev_accruacy_by_layer = np.mean(all_tv_dev_accruacy_by_layer, axis=0)
        std_tv_dev_accruacy_by_layer = np.std(all_tv_dev_accruacy_by_layer, axis=0)

        layers = np.array(list(list(results[model_name].values())[0]["tv_dev_accruacy_by_layer"].keys()))
        layers_fraction = layers / (max(layers) / 0.9)

        x_values = layers
        if normalize_x_axis:
            x_values = x_values / num_layers[model_name]

        # Use different marker for each model and increase the marker size
        plt.plot(
            x_values,
            mean_tv_dev_accruacy_by_layer,
            marker=markers[idx],
            markersize=10,
            label=MODEL_DISPLAY_NAME_MAPPING[model_name],
            alpha=0.8,
        )
        plt.fill_between(
            x_values,
            mean_tv_dev_accruacy_by_layer - std_tv_dev_accruacy_by_layer,
            mean_tv_dev_accruacy_by_layer + std_tv_dev_accruacy_by_layer,
            alpha=0.1,
        )

    plt.xlabel("Layer")
    plt.ylabel("Accuracy")

    plt.ylim(0.0, 1.0)

    # place the legend on the top right corner
    plt.legend(loc="upper right")

    # save the figure
    save_path = os.path.join(FIGURES_DIR, f"accuracy_per_layer{filename_suffix}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

#★★
def plot_accuracy_by_layer_per_task(results, model_names: List[str], task_names: List[str] = None, normalize_x_axis: bool = False, output_dir: str = None):
    """
    各タスクごとに、各モデルの各レイヤーでの精度をプロットする関数
    
    Args:
        results: 実験結果の辞書
        model_names: プロットするモデルの名前リスト
        task_names: プロットするタスクの名前リスト（Noneの場合は最初のモデルのすべてのタスク）
        normalize_x_axis: レイヤー番号を正規化するかどうか
        output_dir: 出力ディレクトリ（Noneの場合はFIGURES_DIR）
    """
    if output_dir is None:
        output_dir = os.path.join(FIGURES_DIR, "per_task_plots")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if task_names is None:
        task_names = list(results[model_names[0]].keys())
    
    # マーカーを定義
    markers = ["o", "^", "s", "P", "X", "D", "v", "*", "p", "h"]
    
    # レイヤー数の定義
    num_layers = {
        "llama_7B": 32,
        "llama_13B": 32,
        # 必要に応じて他のモデルも追加
    }
    
    # 各タスクごとにグラフを作成
    for task_name in task_names:
        plt.figure(figsize=(10, 5))
        plt.rc("font", size=16)
        # plt.title(f"Accuracy by Layer for {task_name.replace('_', ' ').title()}")
        
        # 各モデルをプロット
        for idx, model_name in enumerate(model_names):
            if task_name not in results[model_name]:
                print(f"{model_name}の{task_name}スキップ")
                continue
            
            # レイヤー別の精度データを取得
            task_results = results[model_name][task_name]
            dev_accuracy_by_layer = task_results["tv_dev_accruacy_by_layer"]
            
            # レイヤー番号とその精度値を取得
            layers = np.array(list(dev_accuracy_by_layer.keys()))
            accuracies = np.array(list(dev_accuracy_by_layer.values()))
            
            # X軸の値（必要に応じて正規化）
            x_values = layers
            if normalize_x_axis and model_name in num_layers:
                x_values = x_values / num_layers[model_name]
            
            # プロットを描画
            plt.plot(
                x_values,
                accuracies,
                marker=markers[idx % len(markers)],  
                markersize=10,
                label=MODEL_DISPLAY_NAME_MAPPING.get(model_name, model_name),
                alpha=0.8,
            )
        
        # グラフの装飾
        plt.xlabel("Layer")
        plt.ylabel("Accuracy")
        plt.ylim(0.0, 1.0)
        plt.legend(loc="upper right")
        
        # ファイル名で使用できる安全なタスク名を作成
        safe_task_name = task_name.replace("/", "_")
        
        # 図を保存
        save_path = os.path.join(output_dir, f"accuracy_per_layer_{safe_task_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

def create_results_latex_table(grouped_accuracies_df):
    prefix = "\\onecolumn\n\\begin{center}\n\\small"
    suffix = "\\end{center}\n\\twocolumn"

    caption_and_label = (
        r"""\caption{Complete results of the main experiment for all tasks and models.} \label{table:main_results} \\"""
    )

    table_df = grouped_accuracies_df.copy()

    table_df = table_df.sort_values(by=["model", "task_type", "task_name"])

    table_df["task_name"] = table_df["task_name"].str.replace("_", " ").str.capitalize()
    table_df["task_type"] = table_df["task_type"].str.capitalize()

    table_df.columns = table_df.columns.str.replace("_", " ").str.capitalize()

    # set ["model", "task_type", "task_name"] as index
    table_df = table_df.set_index(["Model", "Task type", "Task name"])

    # sort the index by the model name. make sure to have "Pythia 2.9B" before "Pythia 7B" etc.
    table_df = table_df.sort_index(level=0)

    table_latex = table_df.to_latex(index=True, multirow=True, float_format="%.2f")

    table_latex = table_latex.replace("tabular", "longtable")

    original_head = "\n".join(table_latex.split("\n")[1:5])

    new_head = (
        original_head
        + r"""
    \endfirsthead

    \multicolumn{3}{c}
    {{\bfseries \tablename\ \thetable{} -- continued from previous page}} \\
    """
        + original_head
        + r"""
    \endhead
    """
    )

    table_latex = table_latex.replace(original_head, new_head)

    # add the caption and label after the first line
    table_latex = (
        "\n".join(table_latex.split("\n")[:1])
        + "\n\n"
        + caption_and_label
        + "\n\n"
        + "\n".join(table_latex.split("\n")[1:])
    )

    final_latex = prefix + "\n" + table_latex + "\n" + suffix

    save_path = os.path.join(FIGURES_DIR, "main_experiment_results_table.tex")
    with open(save_path, "w") as f:
        f.write(final_latex)


def create_top_tokens_table(results):
    # Top Tokens

    task_names = [
        "translation_ja_en",
        "translation_en_ja",
        
    ]
    # task_names = [
    #     "translation_fr_en",
    #     "linguistic_present_simple_gerund",
    #     #"knowledge_country_capital",
    #     "algorithmic_next_letter",
    #     # Translation
    #     "translation_es_en",
    #     "translation_en_fr",
    #     "translation_en_es",
    #     # Linguistic
    #     "linguistic_present_simple_past_simple",
    #     "linguistic_plural_singular",
    #     "linguistic_antonyms",
    #     # Knowledge
    #     # "knowledge_person_language",
    #     # "knowledge_location_continent",
    #     # "knowledge_location_religion",
    #     # Algorithmic
    #     "algorithmic_prev_letter",
    #         "algorithmic_list_first",
    #         "algorithmic_list_last",
    #         "algorithmic_to_upper",
    #         "algorithmic_to_lower",
    # ]
    model_names = ["gpt-j_6B", "llama_7B", "llama_13B", "pythia_2.8B","pythia_12B", "swallow_7B", "xalma_13B", "youko_8B", "Qwen_j_14B","Qwen_14B",]

    df_data = {}

    for model_name in model_names:
        df_data[model_name] = {}
        model_results = results[model_name]

        def remove_duplicates_ignore_case(lst):
            seen = set()
            output = []
            for s in lst:
                if s.lower() not in seen:
                    output.append(s)
                    seen.add(s.lower())
            return output

        top_words_per_task = {}
        for task_name in task_names:
            task_results = model_results[task_name]

            dev_accuracy_by_layer = task_results["tv_dev_accruacy_by_layer"]
            best_layer = max(dev_accuracy_by_layer, key=dev_accuracy_by_layer.get) + 2

            top_words = task_results["tv_ordered_tokens_by_layer"][best_layer]

            top_words = [x.strip() for x in top_words]

            # filter tokens that are only a-z or A-Z
            top_words = [w for w in top_words if re.match("^[a-zA-Z]+$", w)]

            # remove duplicates
            top_words = remove_duplicates_ignore_case(top_words)

            # remove short words
            # top_words = [w for w in top_words if len(w) > 1]

            top_words_per_task[task_name] = ", ".join(top_words[:20])

        df_data[model_name] = top_words_per_task

    # create a dataframe with 2 indexes: model and task, and 1 column: top tokens
    df = pd.DataFrame.from_dict(df_data, orient="index").stack().to_frame()

    # save the table as a latex table
    save_path = os.path.join(FIGURES_DIR, "top_tokens_table.tex")
    with open(save_path, "w") as f:
        f.write(df.to_latex())


def create_all_figures(experiment_id: str):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    results = load_main_results(experiment_id)
    accuracies = extract_accuracies(results)
    accuracies_df = create_accuracies_df(results)
    grouped_accuracies_df = create_grouped_accuracies_df(accuracies_df)

    plot_avg_accuracies_per_model(grouped_accuracies_df)
    plot_accuracy_by_layer(results, model_names=["gpt-j_6B", "llama_7B", "llama_13B", "pythia_2.8B","pythia_12B", "swallow_7B", "xalma_13B", "youko_8B", "Qwen_j_14B","Qwen_14B"])
    #plot_accuracy_by_layer(
     #   results, model_names=["pythia_2.8B", "pythia_6.9B", "pythia_12B", "gpt-j_6B"], filename_suffix="_appendix"
   # )
    # create_results_latex_table(grouped_accuracies_df)
    # create_top_tokens_table(results)
def create_all_figures_sample_tasks(experiment_id: str):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    results = load_main_results(experiment_id)
    accuracies = extract_accuracies(results)
    accuracies_df = create_accuracies_df(results)
    grouped_accuracies_df = create_grouped_accuracies_df(accuracies_df)
        # 利用可能なモデル名を確認
    #print("利用可能なモデル名:", list(results.keys()))
    # 既存のグラフをプロット
    plot_avg_accuracies_per_model(grouped_accuracies_df)
    plot_accuracy_by_layer(results, model_names=["gpt-j_6B", "llama_7B", "llama_13B", "pythia_2.8B","pythia_12B", "swallow_7B", "xalma_13B", "youko_8B", "Qwen_j_14B","Qwen_14B"])
    
    # 特定のタスクのみをプロットする場合
    # タスク名とタスクタイプを分けて指定
    specific_task_names = [
        "ja_en",
        "en_ja",
    ]
    specific_task_types = ["translation"]
    
    # タスクごとのレイヤー精度グラフをプロット（こちらは元のタスク名フルネームを使用）
    full_task_names = [f"{task_type}_{task_name}" for task_type in specific_task_types for task_name in specific_task_names]
    plot_accuracy_by_layer_per_task(
        results, 
        model_names=["gpt-j_6B", "llama_7B", "llama_13B", "pythia_2.8B","pythia_12B", "swallow_7B", "xalma_13B", "youko_8B", "Qwen_j_14B","Qwen_14B"],
        task_names=full_task_names
    )
    
    # タスクごとのモデル精度の棒グラフをプロット
    plot_accuracies_per_model_per_task(
        grouped_accuracies_df,
        task_names=specific_task_names,
        task_types=specific_task_types
    )
    
    create_results_latex_table(grouped_accuracies_df)
    create_top_tokens_table(results)

def create_all_figures_grouped_sample_tasks(experiment_id: str):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    results = load_main_results(experiment_id)
    
    # 特定のタスクのみをプロット
    specific_tasks = [
        "translation_ja_en",
        "translation_en_ja",
    ]
    
    # グループ分けしたタスクごとのレイヤー精度グラフをプロット
    plot_accuracy_by_layer_per_task_grouped(
        results, 
        task_names=specific_tasks
    )
