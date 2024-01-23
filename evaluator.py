import torch
from sentence_transformers import SentenceTransformer, util
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import pickle
import re
from utils.normalizor import normalizor
import tqdm
import math
# from angle_emb import AnglE

# angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda() 

#model = SentenceTransformer('all-mpnet-base-v2')
#model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
#model = SentenceTransformer('sentence-t5-large')
#model = SentenceTransformer('BAAI/bge-large-en')
#model = SentenceTransformer('KnutJaegersberg/infoxlm-large-sentence-embeddings')
#model = SentenceTransformer('all-roberta-large-v1')
#model = SentenceTransformer('msmarco-distilbert-base-tas-b')
#model = SentenceTransformer('pritamdeka/S-Scibert-snli-multinli-stsb')
#model = SentenceTransformer('bert-base-uncased')
# model = SentenceTransformer('all-mpnet-base-v2')
model = SentenceTransformer('llmrails/ember-v1')
tree : cKDTree
Comments = []

with open("data.pkl", "rb") as f:
    raw = pickle.load(f)
    tree = pickle.loads(raw)

with open("comments.pkl", "rb") as f:
    Comments = pickle.load(f)

correct = 0
dif = 0
mp = {}
ma = 0

def findFunc(s):
    s = re.sub("#.*", "", s)
    d = re.findall(r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"', s,re.DOTALL | re.MULTILINE)
    for i in d:
        s = s.replace(i, '')
    s = re.sub('.*;', "", s)
    s = re.sub("if(.*)", "", s)
    s = re.sub("for(.*)", "", s)
    s = re.sub("while(.*)", "", s)
    k = re.findall(".*\(.*\)", s)
    ret = ""
    for i in k:
        i = re.sub("\(.*\)", "", i)
        ret += i.split(' ')[-1]
    ret = re.sub(r"[^a-zA-Z]", "", ret)
    return ret

for num, (position, comment) in enumerate(Comments):
    position = position[39:]
    position = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])","",position)
    # s = ""
    # with open(position, 'r', encoding="utf-8") as fp:
    #     for line in fp:
    #         s += line
    # s = findFunc(s)
   # q = tree.query(normalizor(list(angle.encode(position, to_numpy=True)[0])), 5)[1]
    q = tree.query(normalizor(model.encode(position, convert_to_tensor = False)), 5)[1]
    for y in q:
        if abs(y - num) < len(Comments) / 100:
            correct += 1
            break
            
   # dif += (y - num) ** 2
    #ma = max(ma, abs(y - num))
    #print(abs(y - num))
    # if abs(y - num) < len(Comments) / 100:
    #     correct += 1
    # else:
    #     if y in mp:
    #         mp[y] += 1
    #     else:
    #         mp[y] = 1
    #     a, b = Comments[y]
     #   print(f"key:{comment} result:{b}")
    if num % 100 == 1:
        print("[{}/{}]({:.2f}%) correct:{}(rate:{:.2f}%)".format(num, len(Comments), num/len(Comments)*100, correct, correct / (num + 1) * 100))

print('{:.2f}%'.format(correct / len(Comments) * 100))
#print(math.sqrt(dif)/len(Comments))
#print(ma)
# ls = sorted(mp.items(), key=lambda item: item[1])
# ls.reverse()
# for i, (key, value) in enumerate(ls):
#     position, comment = Comments[key]
#     print("num:{}, value:{}, position:{}".format(key, value, position))
#     if i > 20:
#         break