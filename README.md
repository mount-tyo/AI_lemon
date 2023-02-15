# lemon
## 準備

pytorch

```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

その他

```
pip install -r requirements.txt
```

## train_images をラベル毎に分類

dataset/classified/ が作成される

```
python visualize.py
```

## train_imagesから黄色の抽出、エッジ検出をしてレモンの取り出し

preprocessed/ が作成される
preprocessed/ に前処理後の画像が「pp_0000.jpg」の形式で保存される
```
>>python preprocess.py
```