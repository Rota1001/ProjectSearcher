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

#model = SentenceTransformer('all-mpnet-base-v2')
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
#model = SentenceTransformer('msmarco-distilbert-base-tas-b')
#model = SentenceTransformer('pritamdeka/S-Scibert-snli-multinli-stsb')
#model = SentenceTransformer('bert-base-uncased')
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
    x, y = tree.query(normalizor(model.encode(position, convert_to_tensor = False)))
    dif += (y - num) ** 2
    
    #print(abs(y - num))
    if abs(y - num) < 163:
        correct += 1
    else:
        if y in mp:
            mp[y] += 1
        else:
            mp[y] = 1
        a, b = Comments[y]
     #   print(f"key:{comment} result:{b}")
    if num % 10 == 1:
        print("[{}/{}]({:.2f}%) correct:{}(rate:{:.2f}%)".format(num, len(Comments), num/len(Comments)*100, correct, correct / (num + 1) * 100))

print('{:.2f}%'.format(correct / len(Comments) * 100))
print(math.sqrt(dif/len(Comments)))
ls = sorted(mp.items(), key=lambda item: item[1])
ls.reverse()
for key, value in ls:
    position, comment = Comments[key]
    print("num:{}, value:{}, position:{}".format(key, value, position))
