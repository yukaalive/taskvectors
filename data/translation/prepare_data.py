import csv
import os

def prepare_translation_data(input_file="translation_data.csv", 
                           output_file="translation_data_small.csv",
                           num_samples=10000):
    """
    翻訳データの準備：データサイズ削減のみ
    
    Args:
        input_file: 元のCSVファイル名
        output_file: 出力CSVファイル名
        num_samples: 使用するサンプル数
    """
    
    print(f"元のデータファイル: {input_file}")
    print(f"出力ファイル: {output_file}")
    print(f"使用サンプル数: {num_samples}")
    
    # 元のファイルサイズを確認
    if os.path.exists(input_file):
        file_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
        print(f"元のファイルサイズ: {file_size:.2f} MB")
    else:
        print(f"エラー: ファイル {input_file} が見つかりません。")
        return None
    
    print("データを読み込み中...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            
            # ヘッダーを読み込み
            header = next(reader)
            print(f"元のカラム: {header}")
            
            # データを読み込み（最初のnum_samples行のみ）
            data_rows = []
            sample_rows = []  # 最初の5行を表示用に保存
            
            for i, row in enumerate(reader):
                if i >= num_samples:
                    break
                
                data_rows.append(row)
                
                # 最初の5行をサンプルとして保存
                if i < 5:
                    sample_rows.append(row)
            
            print(f"読み込み完了: {len(data_rows)} 行")
            
            # 元のデータの例を表示
            print("\n元のデータの例:")
            for i, row in enumerate(sample_rows):
                print(f"行 {i+1}: {row}")
            
            # データを保存（元の形式のまま）
            with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(header)  # ヘッダーを書き込み
                writer.writerows(data_rows)  # データを書き込み
            
            print(f"\n削減済みデータを保存: {output_file}")
            
            # 出力ファイルサイズを確認
            if os.path.exists(output_file):
                output_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
                print(f"出力ファイルサイズ: {output_size:.2f} MB")
            
            # データ統計を表示
            print(f"\n統計情報:")
            print(f"- 総行数: {len(data_rows)}")
            print(f"- データ削減率: {(1 - len(data_rows) / 2801389) * 100:.1f}%")
            
            return data_rows
            
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return None

def create_sample_sizes():
    """
    異なるサイズのサンプルデータセットを作成
    """
    sizes = [1000, 5000, 10000, 50000]
    
    for size in sizes:
        output_file = f"translation_data_{size}.csv"
        print(f"\n{'='*50}")
        print(f"サイズ {size} のデータセットを作成中...")
        prepare_translation_data(
            input_file="translation_data.csv",
            output_file=output_file,
            num_samples=size
        )

if __name__ == "__main__":
    print("翻訳データの準備を開始します...")
    
    # デフォルト設定で実行（10,000サンプル）
    result = prepare_translation_data()
    
    if result is not None:
        print("\n✅ データ準備が完了しました！")
        print("次のステップ:")
        print("1. sft20250614_fixed.py の data_file を 'translation_data_prepared.csv' に変更")
        print("2. ファインチューニングを再実行")
    else:
        print("\n❌ データ準備に失敗しました。")
    
    # 複数サイズのデータセットを作成する場合（コメントアウト）
    # create_sample_sizes()
