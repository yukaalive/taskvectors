import json  # JSONを扱うためのモジュールをインポート
import os    # ファイルパス操作のためのモジュール

# JSONファイルを読み込む関数
def read_json_file(file_path):
    try:
        # ファイルを開いて内容を読み込む
        with open(file_path, 'r', encoding='utf-8') as file:
            # JSONデータをPythonの辞書（dict）に変換
            data = json.load(file)
            print(f"ファイル '{file_path}' を正常に読み込みました。")
            return data
    except FileNotFoundError:
        print(f"エラー: ファイル '{file_path}' が見つかりません。")
        return None
    except json.JSONDecodeError:
        print(f"エラー: ファイル '{file_path}' は有効なJSONフォーマットではありません。")
        return None
    except Exception as e:
        print(f"エラー: ファイルの読み込み中に問題が発生しました: {str(e)}")
        return None

# JSONファイルを別名で保存する関数
def save_json_file(data, output_file_path):
    try:
        # ファイルを書き込みモードで開く
        with open(output_file_path, 'w', encoding='utf-8') as file:
            # データをJSONとして書き込む（日本語対応・整形あり）
            json.dump(data, file, ensure_ascii=False, indent=2)
            print(f"データを '{output_file_path}' に正常に保存しました。")
            return True
    except Exception as e:
        print(f"エラー: ファイルの保存中に問題が発生しました: {str(e)}")
        return False

# メイン処理
def main():
    # 入力ファイルのパスを指定（ここを自分のファイルパスに変更してください）
    input_file_path = '/home/yukaalive/2025workspace/task_vectors/2_icl_task_vectors/data/translation/en_ja.json'
    
    # ユーザーに出力ファイル名を尋ねる
    output_file_name = '/home/yukaalive/2025workspace/task_vectors/2_icl_task_vectors/data/translation/ja_en.json'
    
    
    # 出力ファイル名が空の場合はデフォルト名を使用
    if not output_file_name:
        output_file_name = "en_ja.json"
        print(f"ファイル名が指定されなかったため、デフォルト名 '{output_file_name}' を使用します。")
    
    # 出力ファイル名に.jsonの拡張子がない場合は追加
    if not output_file_name.lower().endswith('.json'):
        output_file_name += '.json'
        print(f"拡張子が追加されました。ファイル名: '{output_file_name}'")
    
    # 入力ファイルと同じディレクトリに出力ファイルを作成
    input_dir = os.path.dirname(input_file_path)
    output_file_path = os.path.join(input_dir, output_file_name)
    
    # すでに同名のファイルが存在する場合の確認
    if os.path.exists(output_file_path):
        overwrite = input(f"'{output_file_name}' は既に存在します。上書きしますか？（y/n）: ")
        if overwrite.lower() != 'y':
            print("保存をキャンセルしました。")
            return
    
    # JSONファイルを読み込む
    data = read_json_file(input_file_path)
    
    if data is not None:
        # キーと値を入れ替える（必要に応じてこの部分を変更）
        swapped_data = {}
        for key, value in data.items():
            swapped_data[value] = key
        
        # 読み込んだデータの情報を表示
        print(f"\n元のデータ: {len(data)} 項目")
        print(f"変換後のデータ: {len(swapped_data)} 項目")
        
        # 変換後のデータを保存
        save_json_file(swapped_data, output_file_path)

# プログラムを実行
if __name__ == "__main__":
    main()