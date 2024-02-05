# import utils.fileLoader as fileLoader
import fileLoader_with_sentence_comparison as fileLoader
import functools
import pickle
import torch
from sentence_transformers import SentenceTransformer, util
from scipy.spatial import cKDTree
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import re

##加了.h > 0.6

Comments = fileLoader.getComments(input("Please input your project directory:"))

model = SentenceTransformer('all-roberta-large-v1')
# model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

model.cuda()
model.eval()
tmp = []

with open("data/comments.pkl", "wb") as f:
    pickle.dump(Comments, f)
# with open("data/comments.pkl", "rb") as f:
#     Comments = pickle.load(f)

for position, comment in tqdm(Comments):
   # tmp.append(list(angle.encode(comment, to_numpy=True)[0]))
    x = position[39:]
    # x = x.replace("/", " ")
  #  x = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])"," ",x)
    tmp.append(model.encode(x, convert_to_tensor = False))
tmpnor = normalize(tmp, norm='l2')
tree = cKDTree(tmpnor)
# tree = cKDTree(tmp)
raw = pickle.dumps(tree)
with open("data/data.pkl", "wb") as f:
    pickle.dump(raw, f)