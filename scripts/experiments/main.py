# This must be first
from dotenv import load_dotenv

load_dotenv(".env")

import sys
import os
import pickle
import time
from typing import Optional

from transformers import PreTrainedModel, PreTrainedTokenizer

from scripts.utils import MAIN_RESULTS_DIR, main_experiment_results_dir

from core.data.task_helpers import get_all_tasks, get_task_by_name
from core.models.llm_loading import load_model_and_tokenizer
from core.models.utils.inference import hidden_to_logits
from core.analysis.utils import logits_top_tokens
from core.analysis.evaluation import calculate_accuracy_on_datasets
from core.task_vectors import run_icl, run_task_vector
from core.utils.misc import limit_gpus, seed_everything
from core.experiments_config import MODELS_TO_EVALUATE, TASKS_TO_EVALUATE


def get_results_file_path(model_type: str, model_variant: str, experiment_id: str = "") -> str:
    return os.path.join(main_experiment_results_dir(experiment_id), f"{model_type}_{model_variant}.pkl")


def evaluate_task(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, task_name: str, num_examples: int) -> None:
    seed_everything(41)
    accuracies = {}

    task = get_task_by_name(tokenizer=tokenizer, task_name=task_name)

    # Evaluate baseline
    baseline_datasets = task.create_datasets(num_datasets=100, num_examples=0)
    predictions = run_icl(model, tokenizer, task, baseline_datasets, include_train=False)
    accuracies["baseline"] = calculate_accuracy_on_datasets(task, predictions, baseline_datasets)

    # Evaluate ICL and Task Vector
    # TODO: Change back to 400, 100
    # num_test_datasets, num_dev_datasets = 400, 100
    num_test_datasets, num_dev_datasets = 50, 50
    test_datasets = task.create_datasets(num_datasets=num_test_datasets, num_examples=num_examples)
    dev_datasets = task.create_datasets(num_datasets=num_dev_datasets, num_examples=num_examples)
    
    # デバッグ出力：データセットの内容を表示
    print("\n=== Debug: Example Datasets ===")
    print(f"テストデータセット総数: {len(test_datasets)}")
    print(f"開発データセット総数: {len(dev_datasets)}")
    print(f"各データセットのショット数: {num_examples}")
    
    # テストデータセットと開発データセットの重複を確認
    test_inputs = [dataset.test_input for dataset in test_datasets]
    test_outputs = [dataset.test_output for dataset in test_datasets]
    dev_inputs = [dataset.test_input for dataset in dev_datasets]
    dev_outputs = [dataset.test_output for dataset in dev_datasets]
    
    # 重複するテスト入力の数を計算
    duplicates = set(test_inputs) & set(dev_inputs)
    print(f"テストデータセットと開発データセットで重複するテスト入力の数: {len(duplicates)}")
    
    if len(duplicates) > 0:
        print("重複するテスト入力の例（最大5つ）:")
        for i, dup in enumerate(list(duplicates)[:5]):
            print(f"  {i+1}: {dup}")
    
    # すべてのテスト入力と出力をファイルに保存
    debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    with open(os.path.join(debug_dir, f"{task_name}_test_inputs_outputs.txt"), "w") as f:
        f.write("=== テストデータセットのテスト入力と出力 ===\n")
        for i, (inp, out) in enumerate(zip(test_inputs, test_outputs)):
            f.write(f"Dataset {i+1}:\n")
            f.write(f"  Test Input: {inp}\n")
            f.write(f"  Test Output: {out}\n")
        
        f.write("\n=== 開発データセットのテスト入力と出力 ===\n")
        for i, (inp, out) in enumerate(zip(dev_inputs, dev_outputs)):
            f.write(f"Dataset {i+1}:\n")
            f.write(f"  Test Input: {inp}\n")
            f.write(f"  Test Output: {out}\n")
    
    print(f"すべてのテスト入力と出力を {os.path.join(debug_dir, f'{task_name}_test_inputs_outputs.txt')} に保存しました")
    
    # expected_outputsの内容を確認
    print("\n=== expected_outputs の内容 ===")
    print(f"テストデータセットのexpected_outputs: {test_outputs}")
    print(f"開発データセットのexpected_outputs: {dev_outputs}")
    
    # テストデータセットの最初の3つを表示
    for i, dataset in enumerate(test_datasets[:3]):
        print(f"\nTest Dataset {i+1}:")
        print(f"  Test Input: {dataset.test_input}")
        print(f"  Test Output: {dataset.test_output}")
        print(f"  Train Inputs (Shots):")
        for j, train_input in enumerate(dataset.train_inputs):
            print(f"    {j+1}: {train_input}")
        print(f"  Train Outputs:")
        for j, train_output in enumerate(dataset.train_outputs):
            print(f"    {j+1}: {train_output}")
    
    # 開発データセットの最初の3つを表示
    for i, dataset in enumerate(dev_datasets[:3]):
        print(f"\nDev Dataset {i+1}:")
        print(f"  Test Input: {dataset.test_input}")
        print(f"  Test Output: {dataset.test_output}")
        print(f"  Train Inputs (Shots):")
        for j, train_input in enumerate(dataset.train_inputs):
            print(f"    {j+1}: {train_input}")
        print(f"  Train Outputs:")
        for j, train_output in enumerate(dataset.train_outputs):
            print(f"    {j+1}: {train_output}")
    
    print("\n=== End Debug Output ===\n")
    icl_predictions = run_icl(model, tokenizer, task, test_datasets)
    tv_predictions, tv_dev_accuracy_by_layer, task_hiddens = run_task_vector(
        model,
        tokenizer,
        task,
        test_datasets,
        dev_datasets,
    )
    print("tv_dev_by_layer")
    accuracies["tv_dev_by_layer"] = tv_dev_accuracy_by_layer
    accuracies["icl"] = calculate_accuracy_on_datasets(task, icl_predictions, test_datasets)
    accuracies["tv"] = calculate_accuracy_on_datasets(task, tv_predictions, test_datasets)
    tv_ordered_tokens_by_layer = {}
    try:
        for layer_num in tv_dev_accuracy_by_layer.keys():
            task_hidden = task_hiddens.mean(axis=0)[layer_num]
            logits = hidden_to_logits(model, task_hidden)
            tv_ordered_tokens_by_layer[layer_num] = logits_top_tokens(logits, tokenizer, k=100)
    except Exception as e:
        print("Error:", e)

    return accuracies, tv_ordered_tokens_by_layer


def run_main_experiment(
    model_type: str,
    model_variant: str,
    experiment_id: str = "",
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
) -> None:
    print("Evaluating model:", model_type, model_variant)

    results_file = get_results_file_path(model_type, model_variant, experiment_id=experiment_id)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    if os.path.exists(results_file):
        with open(results_file, "rb") as f:
            results = pickle.load(f)
    else:
        results = {}

    limit_gpus(range(0, 8))

    print("Loading model and tokenizer...")
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(model_type, model_variant)
    print("Loaded model and tokenizer.")

    tasks = get_all_tasks(tokenizer=tokenizer)

    num_examples = 5

    for i, task_name in enumerate(TASKS_TO_EVALUATE):
        task = tasks[task_name]
        if task_name in results:
            print(f"Skipping task {i+1}/{len(tasks)}: {task_name}")
            continue
        results[task_name] = {}

        print("\n" + "=" * 50)
        print(f"Running task {i+1}/{len(tasks)}: {task_name}")

        tic = time.time()
        accuracies, tv_ordered_tokens_by_layer = evaluate_task(model, tokenizer, task_name, num_examples)

        print(f"Baseline Accuracy: {accuracies['baseline']:.2f}")
        print(f"ICL Accuracy: {accuracies['icl']:.2f}")
        print(f"Task Vector Accuracy: {accuracies['tv']:.2f}")
        print(f"Dev Accuracy by layer: ", end="")
        for layer, accuracy in accuracies["tv_dev_by_layer"].items():
            print(f"{layer}: {accuracy:.2f}, ", end="")
        print()
        print("Time:", time.time() - tic)

        results[task_name] = {
            "baseline_accuracy": accuracies["baseline"],
            "num_examples": num_examples,
            "icl_accuracy": accuracies["icl"],
            "tv_accuracy": accuracies["tv"],
            "tv_dev_accruacy_by_layer": accuracies["tv_dev_by_layer"],
            "tv_ordered_tokens_by_layer": tv_ordered_tokens_by_layer,
        }

        with open(results_file, "wb") as f:
            pickle.dump(results, f)


def get_new_experiment_id() -> str:
    return str(
        max([int(results_dir) for results_dir in os.listdir(MAIN_RESULTS_DIR) if results_dir.isdigit()] + [0]) + 1
    )


def main():
    if len(sys.argv) == 1:
        # Run all models
        # Calculate the experiment_id as the max experiment_id + 1
        experiment_id = get_new_experiment_id()
        for model_type, model_variant in MODELS_TO_EVALUATE:
            run_main_experiment(model_type, model_variant, experiment_id=experiment_id)
    else:
        if len(sys.argv) == 2:
            model_num = int(sys.argv[1])
            model_type, model_variant = MODELS_TO_EVALUATE[model_num]
        elif len(sys.argv) == 3:
            model_type, model_variant = sys.argv[1:]

        run_main_experiment(model_type, model_variant)


if __name__ == "__main__":
    main()
