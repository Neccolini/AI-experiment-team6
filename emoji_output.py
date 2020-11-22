from utils.config import *
from utils.predict import predict,bert_model
import codecs
import sys
args=sys.argv
file_=sys.stdout
if len(args)>1:
    file_=codecs.open(args[1],"a")
net_trained=bert_model()
print("how many emojis do you want ?")
emoji_num=int(input())
quit=False
while quit==False:
    s=""
    print("Please input the text you want to emojify.")
    input_text=input()
    s+=input_text
    if input_text=="q" or input_text=="\n":
        exit()
    output=predict(input_text, net_trained, emoji_num).tolist()
    for i in range(len(output[0])):
        s+=label_to_emoji[output[0][i]]
    print(s,file=file_)