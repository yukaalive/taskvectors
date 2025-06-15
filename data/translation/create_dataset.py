import json
import pandas as pd
import os
from tqdm import tqdm
import gc
import time

def analyze_large_file(file_path):
    """
    大容量ファイルの構造を分析する
    """
    print(f"=== ファイル分析: {file_path} ===")
    
    if not os.path.exists(file_path):
        print(f"エラー: ファイルが存在しません: {file_path}")
        return None
    
    file_size = os.path.getsize(file_path)
    print(f"ファイルサイズ: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    
    # 文字エンコーディングの自動判定
    encodings = ['utf-8', 'utf-8-sig', 'cp932', 'shift_jis', 'euc-jp', 'latin-1']
    
    for encoding in encodings:
        try:
            print(f"\n--- エンコーディング {encoding} でテスト ---")
            with open(file_path, 'r', encoding=encoding) as f:
                # 最初の1000文字を読んで確認
                sample = f.read(1000)
                lines = sample.split('\n')
                
                print(f"読み込み成功！")
                print(f"最初の5行:")
                for i, line in enumerate(lines[:5]):
                    display_line = line[:100] + ('...' if len(line) > 100 else '')
                    print(f"{i+1:3d}: {display_line}")
                
                # 区切り文字を判定
                tab_count = sample.count('\t')
                comma_count = sample.count(',')
                newline_count = sample.count('\n')
                
                print(f"\n区切り文字分析:")
                print(f"  タブ: {tab_count}")
                print(f"  カンマ: {comma_count}")
                print(f"  改行: {newline_count}")
                
                if tab_count > comma_count:
                    separator = '\t'
                    format_type = 'TSV'
                elif comma_count > tab_count:
                    separator = ','
                    format_type = 'CSV'
                else:
                    separator = None
                    format_type = 'TEXT'
                
                print(f"推定形式: {format_type}")
                
                return {
                    'encoding': encoding,
                    'separator': separator,
                    'format': format_type,
                    'size': file_size
                }
                
        except UnicodeDecodeError:
            print(f"エンコーディング {encoding} では読み込めませんでした")
            continue
        except Exception as e:
            print(f"エラー: {e}")
            continue
    
    print("どのエンコーディングでも読み込めませんでした")
    return None

def count_lines_efficiently(file_path, encoding):
    """
    大容量ファイルの行数を効率的にカウント
    """
    print("行数をカウントしています...")
    line_count = 0
    
    with open(file_path, 'r', encoding=encoding) as f:
        for line in f:
            line_count += 1
            if line_count % 100000 == 0:
                print(f"  {line_count:,} 行処理済み...")
    
    print(f"総行数: {line_count:,}")
    return line_count

def process_large_translation_file(file_path, output_file, batch_size=10000, max_rows=None):
    """
    大容量翻訳ファイルをバッチ処理で変換
    
    Args:
        file_path (str): 入力ファイルのパス
        output_file (str): 出力JSONファイルのパス
        batch_size (int): バッチサイズ
        max_rows (int): 処理する最大行数（Noneの場合は全て処理）
    """
    print(f"\n=== 大容量ファイル処理開始 ===")
    
    # ファイル分析
    file_info = analyze_large_file(file_path)
    if not file_info:
        return None
    
    encoding = file_info['encoding']
    separator = file_info['separator']
    format_type = file_info['format']
    
    print(f"\n使用する設定:")
    print(f"  エンコーディング: {encoding}")
    print(f"  形式: {format_type}")
    separator_name = 'タブ' if separator == '\t' else str(separator)
    print(f"  区切り文字: {separator_name}")
    print(f"  バッチサイズ: {batch_size:,}")
    
    # 行数をカウント
    total_lines = count_lines_efficiently(file_path, encoding)
    
    if max_rows:
        process_lines = min(max_rows, total_lines)
        print(f"処理対象行数: {process_lines:,} / {total_lines:,}")
    else:
        process_lines = total_lines
        print(f"処理対象行数: {process_lines:,}")
    
    # バッチ処理で変換
    all_training_data = []
    processed_count = 0
    valid_count = 0
    
    print(f"\n=== データ変換開始 ===")
    start_time = time.time()
    
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            batch_data = []
            
            for line_num, line in enumerate(f, 1):
                if max_rows and line_num > max_rows:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                # 区切り文字で分割
                if separator:
                    parts = line.split(separator)
                else:
                    # 区切り文字がない場合はスキップ
                    continue
                
                if len(parts) >= 2:
                    english = parts[0].strip()
                    japanese = parts[1].strip()
                    
                    # 有効なデータかチェック
                    if english and japanese:
                        training_data = {
                            "text": f"次の英語を日本語に翻訳してください: {english}",
                            "output": japanese
                        }
                        batch_data.append(training_data)
                        valid_count += 1
                
                processed_count += 1
                
                # バッチ処理
                if len(batch_data) >= batch_size:
                    all_training_data.extend(batch_data)
                    batch_data = []
                    
                    # 進捗表示
                    elapsed = time.time() - start_time
                    progress = processed_count / process_lines * 100
                    print(f"進捗: {processed_count:,}/{process_lines:,} ({progress:.1f}%) "
                          f"有効データ: {valid_count:,} "
                          f"経過時間: {elapsed:.1f}秒")
                    
                    # メモリ管理
                    if len(all_training_data) > 100000:  # 10万件を超えたら一時保存
                        save_batch_to_file(all_training_data, output_file, append=(valid_count > 100000))
                        all_training_data = []
                        gc.collect()
            
            # 残りのバッチを処理
            if batch_data:
                all_training_data.extend(batch_data)
            
            # 最終保存
            if all_training_data:
                append_mode = valid_count > len(all_training_data)
                save_batch_to_file(all_training_data, output_file, append=append_mode)
    
    except Exception as e:
        print(f"処理エラー: {e}")
        return None
    
    elapsed = time.time() - start_time
    print(f"\n=== 処理完了 ===")
    print(f"処理行数: {processed_count:,}")
    print(f"有効データ数: {valid_count:,}")
    print(f"処理時間: {elapsed:.1f}秒")
    print(f"出力ファイル: {output_file}")
    
    return valid_count

def save_batch_to_file(data, output_file, append=False):
    """
    バッチデータをファイルに保存（メモリ効率的）
    """
    if append and os.path.exists(output_file):
        # 既存ファイルに追記する場合
        try:
            with open(output_file, 'r+', encoding='utf-8') as f:
                # 最後の ] を削除
                f.seek(0, 2)  # ファイル末尾に移動
                file_size = f.tell()
                if file_size > 1:
                    f.seek(file_size - 1)
                    last_char = f.read(1)
                    if last_char == ']':
                        f.seek(file_size - 1)
                        f.truncate()
                        
                        # カンマと新しいデータを追加
                        f.write(',\n')
                        for i, item in enumerate(data):
                            if i > 0:
                                f.write(',\n')
                            json.dump(item, f, ensure_ascii=False)
                        f.write('\n]')
                    else:
                        # ファイルが正しい形式でない場合は新規作成
                        append = False
        except:
            # エラーが発生した場合は新規作成
            append = False
    
    if not append:
        # 新規ファイル作成
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('[\n')
            for i, item in enumerate(data):
                if i > 0:
                    f.write(',\n')
                json.dump(item, f, ensure_ascii=False)
            f.write('\n]')

def preview_converted_data(json_file, num_samples=5):
    """
    変換されたデータのプレビュー表示
    """
    print(f"\n=== 変換データプレビュー ===")
    
    try:
        # ファイルサイズをチェック
        file_size = os.path.getsize(json_file)
        print(f"出力ファイルサイズ: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
        
        # 最初の数件だけ読み込み
        with open(json_file, 'r', encoding='utf-8') as f:
            # JSONファイルの最初の部分を読み込み
            content = f.read(50000)  # 最初の50KB
            
        # 不完全なJSONを修正して解析
        if content.endswith(','):
            content = content[:-1] + ']'
        elif not content.endswith(']'):
            content += ']'
        
        try:
            partial_data = json.loads(content)
            print(f"確認できたデータ数: {len(partial_data)}")
            
            for i in range(min(num_samples, len(partial_data))):
                print(f"\n--- サンプル {i+1} ---")
                text_preview = partial_data[i]['text'][:100] + ('...' if len(partial_data[i]['text']) > 100 else '')
                output_preview = partial_data[i]['output'][:100] + ('...' if len(partial_data[i]['output']) > 100 else '')
                print(f"text: {text_preview}")
                print(f"output: {output_preview}")
                
        except json.JSONDecodeError:
            print("JSONの一部を解析できませんでした（データが大きすぎる可能性があります）")
            
    except Exception as e:
        print(f"プレビューエラー: {e}")

def split_large_dataset(input_file, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1):
    """
    大容量データセットを訓練・検証・テスト用に分割
    """
    print(f"\n=== データセット分割 ===")
    
    if abs(train_ratio + dev_ratio + test_ratio - 1.0) > 0.001:
        print("エラー: 分割比率の合計が1.0になりません")
        return
    
    # まずデータ数をカウント
    print("データ数をカウントしています...")
    data_count = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data_count = len(data)
    except json.JSONDecodeError:
        # 大容量ファイルの場合は行数でカウント
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip().startswith('{'):
                    data_count += 1
    
    print(f"総データ数: {data_count:,}")
    
    train_count = int(data_count * train_ratio)
    dev_count = int(data_count * dev_ratio)
    test_count = data_count - train_count - dev_count
    
    print(f"分割予定:")
    print(f"  訓練用: {train_count:,} ({train_ratio*100:.1f}%)")
    print(f"  検証用: {dev_count:,} ({dev_ratio*100:.1f}%)")
    print(f"  テスト用: {test_count:,} ({test_ratio*100:.1f}%)")
    
    # 出力ファイル名
    base_name = input_file.replace('.json', '')
    train_file = f"{base_name}_train.json"
    dev_file = f"{base_name}_dev.json"
    test_file = f"{base_name}_test.json"
    
    # データを読み込み
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 分割
        train_data = data[:train_count]
        dev_data = data[train_count:train_count + dev_count]
        test_data = data[train_count + dev_count:]
        
        # 各データセットを保存
        datasets = [
            (train_data, train_file, "訓練用"),
            (dev_data, dev_file, "検証用"), 
            (test_data, test_file, "テスト用")
        ]
        
        for dataset, filename, name in datasets:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            print(f"{name}データを {filename} に保存 ({len(dataset):,}件)")
            
    except Exception as e:
        print(f"分割エラー: {e}")

# 使用例
if __name__ == "__main__":
    # 指定されたファイルを処理
    input_file = "raw"
    output_file = "translation_training_data.json"
    
    print("=== 大容量翻訳ファイル処理ツール ===")
    
    # ファイルの存在確認
    if not os.path.exists(input_file):
        print(f"エラー: ファイルが存在しません: {input_file}")
        exit(1)
    
    # オプション選択
    print("\n処理オプション:")
    print("1. 全データを処理")
    print("2. テスト用（最初の10万行のみ）")
    print("3. 中規模（最初の100万行のみ）")
    
    choice = input("選択してください (1/2/3): ").strip()
    
    if choice == "2":
        max_rows = 100000
        output_file = "translation_training_data_test.json"
        print(f"テストモード: 最初の{max_rows:,}行のみ処理")
    elif choice == "3":
        max_rows = 1000000
        output_file = "translation_training_data_medium.json"
        print(f"中規模モード: 最初の{max_rows:,}行のみ処理")
    else:
        max_rows = None
        print("全データを処理します")
    
    # 処理実行
    result = process_large_translation_file(
        input_file, 
        output_file, 
        batch_size=10000, 
        max_rows=max_rows
    )
    
    if result:
        # プレビュー表示
        preview_converted_data(output_file)
        
        # データセット分割の提案
        if result > 1000:  # 1000件以上の場合のみ分割を提案
            print(f"\n生成された {output_file} を訓練・検証・テスト用に分割しますか？ (y/n): ", end="")
            split_choice = input().strip().lower()
            
            if split_choice == 'y':
                split_large_dataset(output_file)
    
    print("\n処理完了！")