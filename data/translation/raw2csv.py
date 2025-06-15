import pandas as pd

def convert_to_csv(input_file, output_file):
    data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            parts = line.split('\t')
            english, japanese = parts
            data.append({
                'input': english.strip(),
                'output': japanese.strip()
            })

    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"変換完了: {len(data)}")

convert_to_csv('raw', 'translation_data.csv')