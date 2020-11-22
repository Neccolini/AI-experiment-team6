from utils.bert import get_config, BertModel,BertForEmoji, set_learned_params
from torch import optim,nn
import torch
from utils.dataloader import DataLoader
from utils.train import train_model


train_dl, val_dl, TEXT, dataloaders_dict= DataLoader(max_length=256, batch_size=32)
# モデル設定のJOSNファイルをオブジェクト変数として読み込む
config = get_config(file_path="./weights/bert_config.json")

# ベースのBERTモデルを生成
net_bert = BertModel(config)

# BERTモデルに学習済みパラメータセット
net_bert = set_learned_params(
    net_bert, weights_path="./weights/pytorch_model.bin")

net = BertForEmoji(net_bert)

# 訓練モードに設定
net.train()

# 勾配計算を最後のBertLayerモジュールと追加した分類アダプターのみ実行

for name, param in net.named_parameters():
    param.requires_grad = False

for name, param in net.bert.encoder.layer[-1].named_parameters():
    param.requires_grad = True

for name, param in net.cls.named_parameters():
    param.requires_grad = True

optimizer = optim.AdamW([
    {'params': net.bert.encoder.layer[-1].parameters(), 'lr': 5e-5},
    {'params': net.cls.parameters(), 'lr': 5e-5}
])

# 損失関数の設定
criterion = nn.CrossEntropyLoss()

# 学習・検証を実行する。
num_epochs = 8   #適宜エポック数は変更してください。
net_trained = train_model(net, dataloaders_dict,
                          criterion, optimizer, num_epochs=num_epochs)


save_path = './weights/bert_fine_tuning_cheat.pth'
torch.save(net_trained.state_dict(), save_path)