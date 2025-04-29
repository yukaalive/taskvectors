- llm_loading.py
    - モデル修正
    - gpu修正
    - 0除算の修正

## タスク別精度

### タスクカテゴリー別平均精度:
- 同じカテゴリー内のタスク精度を平均
    - 例：翻訳タスク平均 = (translation_fr_en + translation_es_en + translation_en_fr + translation_en_es) / 4
- Task Vector精度は、各タスクの最高精度レイヤーの値を使用して平均計算

### 全体平均精度:
- 3つのカテゴリー（翻訳、言語学的、アルゴリズム的）の平均
- 全体平均 = (翻訳タスク平均 + 言語学的タスク平均 + アルゴリズム的タスク平均) / 3
```
export PYTHONPATH=$PYTHONPATH:/home/yukaalive/2025workspace/task_vectors/2_icl_task_vectors
python scripts/figures/main.py
```