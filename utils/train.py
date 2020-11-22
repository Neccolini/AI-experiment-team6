
import time
import torch 
import codecs
import sys

def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs,file=codecs.open("output.txt","a")):
    #file=sys.stdout
    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device,file=file)
    print('-----start-------',file=file)

    # ネットワークをGPUへ
    net.to(device)

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # ミニバッチのサイズ
    batch_size = dataloaders_dict["train"].batch_size
    
    # epochのループ
    for epoch in range(num_epochs):
        print("Train start!")
        # epochごとの訓練と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
            else:
                net.eval()   # モデルを検証モードに
            epoch_loss = 0.0  # epochの損失和
            epoch_corrects = 0  # epochの正解数
            iteration = 1
            # 開始時刻を保存
            t_epoch_start = time.time()
            t_iter_start = time.time()
            # データローダーからミニバッチを取り出すループ
            for batch in (dataloaders_dict[phase]):
                # batchはTextとLableの辞書型変数
                # GPUが使えるならGPUにデータを送る
                inputs = batch.Text[0].to(device) # 文章
                labels = batch.Label.to(device) # ラベル
                # optimizerを初期化
                optimizer.zero_grad()
                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs, token_type_ids=None, attention_mask=None,
                                  output_all_encoded_layers=False, attention_show_flg=False)
                    loss = criterion(outputs, labels)  # 損失を計算

                    #_, preds = torch.max(outputs, 1)  # ラベルを予測
                    _, preds=torch.topk(outputs,2)
                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        if (iteration % 10 == 0):  # 10iterに1度、lossを表示
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            #acc = (torch.sum(preds == labels.data)
                            #       ).double()/batch_size
                            acc=0
                            for i in range(len(labels.data)):
                                print("ans:{} pred:{}".format(labels.data[i],preds[i]),file=codecs.open("check.txt","a"))
                                acc += (labels.data[i] in preds[i])
                            acc=float(acc)/batch_size
                            #acc=(torch.sum(labels.data in preds)).double()/batch_size
                            print('イテレーション {} || Loss: {:.4f} || 10iter: {:.4f} sec. || 本イテレーションの正解率：{}'.format(
                                iteration, loss.item(), duration, acc),file=file)
                            t_iter_start = time.time()

                    iteration += 1
                    # 損失と正解数の合計を更新
                    epoch_loss += loss.item() * batch_size
                    #epoch_corrects += torch.sum(preds == labels.data)
                    for i in range(len(labels.data)):
                        epoch_corrects += (labels.data[i] in preds[i])
            # epochごとのlossと正解率
            t_epoch_finish = time.time()
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = float(epoch_corrects) / len(dataloaders_dict[phase].dataset)
            print('Epoch {}/{} | {:^5} |  Loss: {:.4f} Acc: {:.4f}'.format(epoch+1, num_epochs,
                                                                           phase, epoch_loss, epoch_acc),file=file)
            t_epoch_start = time.time()

    return net