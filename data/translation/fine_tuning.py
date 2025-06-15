import csv, json, mojimoji, re, os, sys, emoji, pickle
import numpy as np
from tqdm import tqdm
from pykakasi import kakasi
#torchの読み込み前に環境変数を固定
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, AutoModelWithLMHead



class Zupposhi_maker:

  gpt_model_name = "cyberagent/DeepSeek-R1-Distill-Qwen-14B-Japanese"

  #データセットのcsvファイル
  csv_path = "/home/yukaalive/2025workspace/task_vectors/4_icl_task_vectors/data/translation/translation_data.csv"
  csv_enc = "utf-8-sig"

  #教師データをGPT用のファイルとして出力する際のパス
  gpt2train_path = "train_data.txt" 

  #文章の最大長
  min_len = 32
  max_len = 100

  #学習回数，バッチサイズ
  Nepo = 100
  bsize = 8
  
  #途中経過を表示する学習回数の間隔
  logging_steps = 200

  #モデルを保存する間隔
  save_freq = 100000

  #結果の出力先
  odir = "/output/"

  #予測時のパラメータ
  top_k = 40 #top-k検索の閾値
  top_p = 1 #top-pの閾値
  num_text = 1 #出力する文の数
  temp = 1.0
  repeat_ngram_size = 1

  #推論にCPUを使用するか
#   use_cpu = True

  def __init__(self, ft_path = None, isTrain = True):
    """コンストラクタ

      コンストラクタ。モデルをファイルから読み込む場合と，
      新規作成する場合で動作を分ける．

      Args:
          ft_path : ファインチューニングされたモデルのパス．
                    Noneを指定すると
          train : 学習を行うか
      Returns:
          なし
    """
    # print("GPU is available : {}".format(torch.cuda.is_available()))

    #モデルの設定
    self.__SetModel(ft_path)

    #教師データの読み込み
    if isTrain:
      self.__LoadDataSet()


  def __SetModel(self, ft_path = None):
    """GPT2の設定

      GPT2のTokenizerおよびモデルを設定する．
      ユーザー定義後と顔文字も語彙として認識されるように設定する．
      
      Args:
          ft_path : ファインチューニング済みのモデルを読み込む
                    何も指定しないとself.gpt_model_nameの事前学習モデルを
                    ネットからダウンロードする．
      Returns:
          なし
    """
    #GPT2のTokenizerのインスタンスを生成
    self.tokenizer = AutoTokenizer.from_pretrained(
              self.gpt_model_name
    )
    #self.tokenizer.do_lower_case = True # 今回はrinna社のモデルではないので必要なし。

    #モデルの読み込み
    if ft_path is not None:
      self.model = AutoModelForCausalLM.from_pretrained(ft_path, device_map="auto")
    else:
      self.model = AutoModelForCausalLM.from_pretrained(self.gpt_model_name, device_map="auto")
  


  def __LoadDataSet(self):
      """データセットのロード
      
          怪文書データセットの読み込み

          Args:
              csv_name (string) : csvファイルのファイル名
              Rtest (float) : テスト用のデータの割合
          Returns:
              なし
      """
          
      #csvファイルを読み込む
      data = []
      with open(self.csv_path, "r", encoding = self.csv_enc) as f:
          reader = csv.reader(f, delimiter = ",")
          for row in reader:
            data.append([row[0], row[1]])
            print(row[0], row[1])
      
      
      #教師データの成形と，絵文字の抽出
      with open(self.gpt2train_path, "w", encoding = "utf-8-sig") as f:
        for row in tqdm(data):
          ret = self.__TextCleaning(row)
          To, Body, From = ret[0], ret[1], ret[2]
          
          #手紙の宛名+送り主から本文を予測するタスクを行う．
          #もしも送り主が空欄でなくて、かつ末尾に句読点や感嘆符が付いていないなら"。"をつける
          if (From != "") & (not (From.endswith( ("．", ".", "。", "!", "！", "?", "？") ))):
            From = From + "。" #末尾に"。"や"！", "?"が付いていないなら区切る。
          if (Body != "") & (not (Body.endswith( ("．", ".", "。", "!", "！", "?", "？") ))):
            Body = Body + "。" #末尾に"。"や"！", "?"が付いていないなら区切る。
          if (To != "") & (not (To.endswith(   ("．", ".", "。", "!", "！", "?", "？") ))):
            To = To + "。" #末尾に"。"や"！", "?"が付いていないなら区切る。
          
          #テキストを学習用の形式に編集
          text = To +  Body + From

          #text = "".join(tokens).replace('▁', '')
          print(text)
          f.write(text + "\n")
  


#   def __TextCleaning(self, texts):
#       """テキストの前処理をする

#         テキストの前処理を行う．具体的に行うこととしては．．．
#         ・全角/半角スペースの除去
#         ・半角数字/アルファベットの全角化
#       """
#       #半角スペース，タブ，改行改ページを削除
#       texts = [re.sub("[\u3000 \t \s \n]", "", t) for t in texts]

#       #半角/全角を変換
#       texts = [mojimoji.zen_to_han(t, kana=False) for t in texts]
#       return texts
  


  def TrainGPT2(self):
    """GPT2のファインチューニング

      GPT2の設定とファインチューニングをする
    """
    #データセットの設定
    train_dataset = TextDataset(
                      tokenizer = self.tokenizer,
                      file_path = self.gpt2train_path,
                      block_size = self.max_len #文章の長さを揃える必要がある
    )

    #データ入力についての設定
    data_collator = DataCollatorForLanguageModeling(
                      tokenizer=self.tokenizer,
                      mlm= False
    )

    #学習についての設定
    os.makedirs(self.odir + "gpt2-ft",  exist_ok=True) #結果出力先のディレクトリがあれば作成
    training_args = TrainingArguments(
                      output_dir=self.odir + "gpt2-ft", 
                      overwrite_output_dir=True,
                      num_train_epochs=self.Nepo,
                      per_device_train_batch_size=self.bsize, 
                      logging_steps=self.logging_steps,
                      save_steps=self.save_freq
    )

    #上記の設定をtransformerのTrainerクラスに適用
    trainer = Trainer(
                      model =self.model,
                      args=training_args,
                      data_collator = data_collator,
                      train_dataset = train_dataset
    )

    #学習開始
    print("start ... ")
    trainer.train()
    print("finish!")
    
    print("saving...")
    #モデルをCPU/GPUのどちらかに移す
    if self.use_cpu: #推論時にCPUの利用を強制する場合の処理
      device = torch.device('cpu')
    else: #特に指定が無いなら，GPUがあるときはGPUを使い，CPUのみの場合はCPUを使う
      device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    self.model.to(device)

    #モデルを保存する
    trainer.save_model()
    print("finish!")
  

    
  def GenLetter(self, prompt):
    """怪文書の生成

      GPT2で怪文書を生成する．
      promptに続く文章を生成して出力する

      Args:
          prompt : 文章の先頭
      Retunrs:
          生成された文章のリスト
    """
    #文章をtokenizerでエンコード
    x = self.tokenizer.encode(prompt, return_tensors="pt")

    if self.use_cpu: #CPUの利用を強制する場合の処理
      device = torch.device('cpu')
    else: #特に指定が無いなら，GPUがあるときはGPUを使い，CPUのみの場合はCPUを使う
      device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    x = x.to(device)
    #gptによる推論
    with torch.no_grad():
      y = self.model.generate(
                        x,
                        min_length=self.min_len,  # 文章の最小長
                        max_length=self.max_len,  # 文章の最大長
                        do_sample=True,   # 次の単語を確率で選ぶ
                        top_k=self.top_k, # Top-Kサンプリング
                        top_p=self.top_p,  # Top-pサンプリング
                        temperature=self.temp,  # 確率分布の調整
                        no_repeat_ngram_size = self.repeat_ngram_size, #同じ単語を何回繰り返していいか
                        num_return_sequences=self.num_text,  # 生成する文章の数
                        pad_token_id=self.tokenizer.pad_token_id,  # パディングのトークンID
                        bos_token_id=self.tokenizer.bos_token_id,  # テキスト先頭のトークンID
                        eos_token_id=self.tokenizer.eos_token_id,  # テキスト終端のトークンID
                        early_stopping=True
                      )
    
    # 特殊トークンをスキップして推論結果を文章にデコード
    res = self.tokenizer.batch_decode(y, skip_special_tokens=True)
    return res