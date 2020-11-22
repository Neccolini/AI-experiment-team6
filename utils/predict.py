import mojimoji
import re
import os
import string
import torch
import pickle
import torchtext
from utils.bert import get_config, load_vocab, BertModel, BertTokenizer,BertForEmoji,set_learned_params
from utils.config import * 
from utils.dataloader import DataLoader

def preprocessing_text(text):
    # 半角・全角の統一
    text = mojimoji.han_to_zen(text) 
    # 改行、半角スペース、全角スペースを削除
    text = re.sub('\r', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('　', '', text)
    text = re.sub(' ', '', text)
    # 数字文字の一律「0」化
    text = re.sub(r'[0-9 ０-９]+', '0', text)  # 数字

    # カンマ、ピリオド以外の記号をスペースに置換
    for p in string.punctuation:
        if (p == ".") or (p == ","):
            continue
        else:
            text = text.replace(p, " ")

    return text



# 前処理と単語分割をまとめた関数を定義
def tokenizer_with_preprocessing(text):
    tokenizer_bert = BertTokenizer(vocab_file=VOCAB_FILE, do_lower_case=False)
    text = preprocessing_text(text)
    ret = tokenizer_bert.tokenize(text)  
    return ret

def pickle_dump(TEXT,path):
    with open(path, "wb") as f:
        pickle.dump(TEXT,f)


def pickle_load(path):
    with open(path, "rb") as f:
        TEXT=pickle.load(f)
    return TEXT


def create_vocab_text():
    TEXT = torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True,
                            lower=False, include_lengths=True, batch_first=True, fix_length=max_length, init_token="[CLS]", eos_token="[SEP]", pad_token='[PAD]', unk_token='[UNK]')
    LABEL = torchtext.data.Field(sequential=False, use_vocab=False)
    train_val_ds, test_ds = torchtext.data.TabularDataset.splits(
        path=DATA_PATH, train='train_dumy.csv',
        test='test_dumy.csv', format='csv',
        fields=[('Text', TEXT), ('Label', LABEL)])
    vocab_bert, ids_to_tokens_bert = load_vocab(vocab_file=VOCAB_FILE)
    TEXT.build_vocab(train_val_ds, min_freq=1)
    TEXT.vocab.stoi = vocab_bert
    pickle_dump(TEXT, PKL_FILE)

    return TEXT


def bert_model():
    if os.path.exists(PKL_FILE)==False:
        create_vocab_text()
    config=get_config(file_path=BERT_CONFIG)
    net_bert=BertModel(config)
    net_trained=BertForEmoji(net_bert)
    net_trained.load_state_dict(torch.load(MODEL_FILE,map_location=torch.device('cpu')))
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net_trained.eval()
    net_trained.to(device)
    return net_trained


def predict(input_text, net_trained,candidate_num=3,output_print=False):
    TEXT=pickle_load(PKL_FILE)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer_bert=BertTokenizer(vocab_file=VOCAB_FILE, do_lower_case=False)
    text=preprocessing_text(input_text)
    text=tokenizer_bert.tokenize(text)
    text.insert(0,"[CLS]")
    text.append("[SEP]")
    token_ids=torch.ones((max_length)).to(torch.int64)
    ids_list=list(map(lambda x:TEXT.vocab.stoi[x],text))
    for i, index in enumerate(ids_list):
        token_ids[i]=index
    ids_list=token_ids.unsqueeze_(0)
    input=ids_list.to(device)
    input_mask=(input != 1)
    outputs, attention_probs=net_trained(input, token_type_ids=None, attention_mask=None, 
                    output_all_encoded_layers=False, attention_show_flg=True)
    
    offset_tensor = torch.tensor(offset,device=device)
    outputs-=offset_tensor
    if output_print==True:print(outputs)
    _, preds=torch.topk(outputs,candidate_num)
    return preds
