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
# model = SentenceTransformer('pritamdeka/S-Scibert-snli-multinli-stsb')
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

for num, (position, comment) in enumerate(Comments):
    position = position[39:]
    position = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])","",position)
    if(num == 0):
        print(position)
    x, y = tree.query(normalizor(model.encode(position, convert_to_tensor = False)))
    dif += (y - num) ** 2
    print(abs(y - num))
    if abs(y - num) < 163:
        correct += 1
    if num % 10 == 1:
        print("[{}/{}]({:.2f}%) correct:{}(rate:{:.2f}%)".format(num, len(Comments), num/len(Comments)*100, correct, correct / num * 100))

print('{:.2f}%'.format(correct / len(Comments) * 100))
print(math.sqrt(dif/len(Comments)))