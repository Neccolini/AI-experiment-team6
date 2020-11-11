from utils.bert import get_config, BertModel,BertForchABSA, set_learned_params
from torch import optim,nn
import torch
from utils.dataloader import get_chABSA_DataLoaders_and_TEXT
from utils.bert import BertTokenizer
train_dl, val_dl, TEXT, dataloaders_dict= get_chABSA_DataLoaders_and_TEXT(max_length=256, batch_size=32)
# モデル設定のJOSNファイルをオブジェクト変数として読み込む
config = get_config(file_path="./weights/bert_config.json")

# ベースのBERTモデルを生成
net_bert = BertModel(config)

# BERTモデルに学習済みパラメータセット
net_bert = set_learned_params(
    net_bert, weights_path="./weights/pytorch_model.bin")

# BERTネガポジ分類モデルを生成(モデルの末尾にネガポジ分類のための全結合層（Linear)を追加)
net = BertForchABSA(net_bert)

# 訓練モードに設定
net.train()

# 勾配計算を最後のBertLayerモジュールと追加した分類アダプターのみ実行

# 1. 全体の勾配計算Falseにセット
for name, param in net.named_parameters():
    param.requires_grad = False

# 2. 最後のBertLayerモジュールだけ勾配計算ありに変更
for name, param in net.bert.encoder.layer[-1].named_parameters():
    param.requires_grad = True

# 3. 識別器（ネガティブorポジティブ）を勾配計算ありに変更
for name, param in net.cls.named_parameters():
    param.requires_grad = True

# BERTの元の部分はファインチューニング
optimizer = optim.Adam([
    {'params': net.bert.encoder.layer[-1].parameters(), 'lr': 5e-5},
    {'params': net.cls.parameters(), 'lr': 5e-5}
], betas=(0.9, 0.999))

# 損失関数の設定
criterion = nn.CrossEntropyLoss()
# nn.LogSoftmax()を計算してからnn.NLLLoss(negative log likelihood loss)を計算


# 学習・検証を実施
from utils.train import train_model

# 学習・検証を実行する。
num_epochs = 1   #適宜エポック数は変更してください。
net_trained = train_model(net, dataloaders_dict,
                          criterion, optimizer, num_epochs=num_epochs)


# 学習したネットワークパラメータを保存(今回は22epoch回した結果を保存する想定でファイル名を記載）
save_path = './weights/bert_fine_tuning_chABSA_22epoch.pth'
torch.save(net_trained.state_dict(), save_path)