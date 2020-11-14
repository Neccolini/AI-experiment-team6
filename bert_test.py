import sys
import pandas as pd
import sentencepiece as spm
import logging
import numpy as np

from keras import utils
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras_bert import load_trained_model_from_checkpoint
from keras_bert import get_custom_objects
from sklearn.metrics import classification_report, confusion_matrix


sys.path.append('modules')

# SentencePieceProccerモデルの読込
spp = spm.SentencePieceProcessor()
spp.Load('wiki-ja.model')
# BERTの学習したモデルの読込
#model_filename = 'models/knbc_finetuning.model' #名前変えてね！！！！！！！！
model_filename = 'models/bert_epo10.model'
model = load_model(model_filename, custom_objects=get_custom_objects())


SEQ_LEN = 204
maxlen = SEQ_LEN

def _get_indice(feature):
    indices = np.zeros((maxlen), dtype=np.int32)

    tokens = []
    tokens.append('[CLS]')
    tokens.extend(spp.encode_as_pieces(feature))
    tokens.append('[SEP]')

    for t, token in enumerate(tokens):
        if t >= maxlen:
            break
        try:
            indices[t] = spp.piece_to_id(token)
        except:
            logging.warn('unknown')
            indices[t] = spp.piece_to_id('<unk>')
    return indices

feature = "あまりうまくいってませんね"

test_features = []
test_features.append(_get_indice(feature))
test_features = np.array(test_features)
test_segments = np.zeros((len(test_features), maxlen), dtype=np.float32)
#print(test_features)
#print(len(test_features[0]))
#print('ahahahahahahh')
#print(test_segments)
#print(len(test_segments[0]))

#print(model.predict([test_features, test_segments]))
predicted_test_labels = model.predict([test_features, test_segments]).argmax(axis=1)

label_data = pd.read_csv('label_id/id_category_twitter.csv')
#print(label_data)
label = label_data.query('id == {}'.format(predicted_test_labels[0]))
label = label.iloc[0]
label_name = label['label']
print(label_name)