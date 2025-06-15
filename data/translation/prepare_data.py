import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('translation_data.csv')

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)


train_df.to_csv('train.csv', index=False, encoding='utf-8')
dev_df.to_csv('dev.csv', index=False, encoding='utf-8')
test_df.to_csv('test.csv', index=False, encoding='utf-8')

print(f"訓練データ: {len(train_df)}行")
print(f"開発データ: {len(dev_df)}行") 
print(f"テストデータ: {len(test_df)}行")