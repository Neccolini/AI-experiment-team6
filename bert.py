#事前学習モデルはhttps://yoheikikuta.github.io/bert-japanese/から
#https://tech-blog.cloud-config.jp/2020-02-06-category-classification-using-bert/

#trainのmax_token_number:204
#testのmax_token_number:188
#f文字列が最新のpythonしか使えないが、pythonアップデート出来なさそう




# BERTの読み込み
#from keras_bert import AdamWarmup, calc_train_steps
#from keras_bert import get_custom_objects
from keras_bert import load_trained_model_from_checkpoint

config_path = 'bert_config.json'
checkpoint_path = 'model.ckpt-1400000'

# 最大のトークン数
SEQ_LEN = 204   # データによって、変えて！！！！！！！！！
BATCH_SIZE = 16
BERT_DIM = 768
LR = 1e-4
# 学習回数    
EPOCH = 1   # 変えて！！！！！！！！！

#bert = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True, trainable=True, seq_len=SEQ_LEN)
bert = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True,  trainable=True, seq_len=SEQ_LEN)
bert.summary()







#以下はBERTモデルの末尾にネガポジ分類のための全結合層を追加している？
# 分類問題用にモデルの再構築
#inputs = bert.inputs[:2]
#dense = bert.get_layer('NSP-Dense').output
#outputs = keras.layers.Dense(units=2, activation='softmax')(dense)
#
#decay_steps, warmup_steps = calc_train_steps(train_y.shape[0],
#    batch_size=BATCH_SIZE, epochs=EPOCHS)
#
#model = keras.models.Model(inputs, outputs)
#model.compile(AdamWarmup(decay_steps=decay_steps, warmup_steps=warmup_steps, lr=LR),
#    loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
#model.summary()

#複数の分類問題なのでhttps://www.inoue-kobo.com/ai_ml/keras-bert/index.htmlを参考にしてみる。









#BERTで特徴量の抽出
import pandas as pd
import sentencepiece as spm
from keras import utils
import logging
import numpy as np

maxlen = 204


sp = spm.SentencePieceProcessor()
sp.Load('wiki-ja.model')

#_get_indice関数:SentencePieceとwikipediaモデルを使用し文章のベクトル化を行っています
def _get_indice(feature):
    indices = np.zeros((maxlen), dtype = np.int32)

    tokens = []
    tokens.append('[CLS]')
    tokens.extend(sp.encode_as_pieces(feature))
    tokens.append('[SEP]')

    for t, token in enumerate(tokens):
        if t >= maxlen:
            break
        try:
            indices[t] = sp.piece_to_id(token)
        except:
            #logging.warn(f'{token} is unknown.')
            logging.warn('{} is unknown.'.format(token))
            indices[t] = sp.piece_to_id('<unk>')

    return indices

#_load_labeldata関数:学習データを読込、_get_indice関数を用いて特徴量を抽出しています。
def _load_labeldata(train_dir, test_dir):
    #train_features_df = pd.read_csv(f'{train_dir}/train_features.csv')
    train_features_df = pd.read_csv('train_features.csv')
    #train_labels_df = pd.read_csv(f'{train_dir}/train_labels.csv')
    train_labels_df = pd.read_csv('train_labels.csv')
    #test_features_df = pd.read_csv(f'{test_dir}/test_features.csv')
    test_features_df = pd.read_csv('test_features.csv')
    #test_labels_df = pd.read_csv(f'{test_dir}/test_labels.csv')
    test_labels_df = pd.read_csv('test_labels.csv')
    label2index = {k: i for i, k in enumerate(train_labels_df['label'].unique())}
    index2label = {i: k for i, k in enumerate(train_labels_df['label'].unique())}
    class_count = len(label2index)
    #昔のAPI keras.utils.np_utils.to_categorical
    #今のAPI keras.utils.to_categorical
    train_labels = utils.to_categorical([label2index[label] for label in train_labels_df['label']], num_classes=class_count)
    test_label_indices = [label2index[label] for label in test_labels_df['label']]
    test_labels = utils.to_categorical(test_label_indices, num_classes=class_count)

    train_features = []
    test_features = []

    for feature in train_features_df['feature']:
        train_features.append(_get_indice(feature))
    train_segments = np.zeros((len(train_features), maxlen), dtype = np.float32)
    for feature in test_features_df['feature']:
        test_features.append(_get_indice(feature))
    test_segments = np.zeros((len(test_features), maxlen), dtype = np.float32)

    #print(f'Trainデータ数: {len(train_features_df)}, Testデータ数: {len(test_features_df)}, ラベル数: {class_count}')
    print('Trainデータ数: {}, Testデータ数: {}, ラベル数: {}'.format(len(train_features_df),len(test_features_df),class_count))

    return {
        'class_count': class_count,
        'label2index': label2index,
        'index2label': index2label,
        'train_labels': train_labels,
        'test_labels': test_labels,
        'test_label_indices': test_label_indices,
        'train_features': np.array(train_features),
        'train_segments': np.array(train_segments),
        'test_features': np.array(test_features),
        'test_segments': np.array(test_segments),
        'input_len': maxlen
    }





#モデル作成関数
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Flatten, GlobalMaxPooling1D
from keras_bert.layers import MaskedGlobalMaxPool1D
from keras import Input, Model
from keras_bert import AdamWarmup, calc_train_steps

def _create_model(input_shape, class_count):
    decay_steps, warmup_steps = calc_train_steps(
        input_shape[0],
        batch_size=BATCH_SIZE,
        epochs=EPOCH,
    )

    bert_last = bert.get_layer(name='NSP-Dense').output
    x1 = bert_last
    output_tensor = Dense(class_count, activation='softmax')(x1)
    # Trainableの場合は、Input Masked Layerが3番目の入力なりますが、
    # FineTuning時には必要無いので1, 2番目の入力だけ使用します。
    # Trainableでなければkeras-bertのModel.inputそのままで問題ありません。
    model = Model([bert.input[0], bert.input[1]], output_tensor)
    model.compile(loss='categorical_crossentropy',
                  optimizer=AdamWarmup(decay_steps=decay_steps, warmup_steps=warmup_steps, lr=LR),
                  #optimizer='nadam',
                  metrics=['mae', 'mse', 'acc'])

    return model






#データのロードとモデルの準備
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
trains_dir = 'data/trains'
tests_dir = 'data/tests'

data = _load_labeldata(trains_dir, tests_dir)
model_filename = 'models/bert_epo1.model'  ###名前を変えてね！！！！！！！！！
model = _create_model(data['train_features'].shape, data['class_count'])
model.summary()




#学習の実行
history = model.fit([data['train_features'], data['train_segments']],
          data['train_labels'],
          epochs = EPOCH,
          batch_size = BATCH_SIZE,
          validation_data=([data['test_features'], data['test_segments']], data['test_labels']),
          shuffle=False,
          verbose = 1,
          #callbacksをコメントアウトすると動く
          #と思ったらなぜか市内でも動くようになった?
          #並列処理できないから?いやできるっぽい、マジでなんでだよ。

          callbacks = [
              ModelCheckpoint(monitor='val_acc', mode='max', filepath=model_filename, save_best_only=True)
          ])



#評価
from IPython.core.display import display
df = pd.DataFrame(history.history)
display(df)


#モデル評価
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model
from keras_bert import get_custom_objects

model = load_model(model_filename, custom_objects=get_custom_objects())
#print(data['test_features'])
#print(len(data['test_features'][0]))
#print('ahahahahahahh')
#print(data['test_segments'])
#print(len(data['test_segments'][0]))
np.set_printoptions(threshold=1000)
#predicted_test_labels = model.predict([data['test_features'], data['test_segments']]).argmax(axis=1) #予測データ
#上位4つまで表示させたい
predicted_test_labels = model.predict([data['test_features'], data['test_segments']])
predicted_test_labels_new = []
for i in range(len(predicted_test_labels)):
    predicted_test_labels_new.append(predicted_test_labels[0][i].argsort()[::-1])
#print(model.predict([data['test_features'], data['test_segments']]))
#print(predicted_test_labels)

numeric_test_labels = np.array(data['test_labels']).argmax(axis=1) #正解データ
#print(np.array(data['test_labels']))
#print(numeric_test_labels)

#上位4つまで表示させるようにしたい
#rate = 0
#count = 0
#for i in range(len(predicted_test_labels_new)):
#    count += 1
#    if numeric_test_labels in predicted_test_labels_new[i][:4]:
#        rate += 1
#print(rate/count)

#スライドの分担決める
#,区切りのデータを使えるようにする → そのデータ自体を整形する → 新しいデータセットのtokenget
#githubにコードあげる

report = classification_report(
        numeric_test_labels, predicted_test_labels, target_names=['blush','cry','devil','flushed','grin','heart_eyes','innocent','laughing','look','rage','scream','sleepy','smile','sob','stuck_out_tongue_closed_eyes','sweat_drops','sweat_smile','tired','triumph','yummy'], output_dict=True)

display(pd.DataFrame(report).T)