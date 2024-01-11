import fileLoader
import functools
import pickle
import torch
from sentence_transformers import SentenceTransformer, util
from scipy.spatial import cKDTree
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


#FileLoader.Download(input("Please input your github url:"))
#Comments = FileLoader.GetComments('./getFile')
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# model = SentenceTransformer('pritamdeka/S-Scibert-snli-multinli-stsb')
#model = SentenceTransformer('all-mpnet-base-v2')
#model.load_state_dict(torch.load('model.pt'))
Comments = fileLoader.getComments(input("Please input your project directory:"))
model.cuda()
#model.eval()
tmp = []
with open("comments.pkl", "wb") as f:
    pickle.dump(Comments, f)
for position, comment in tqdm(Comments):
    tmp.append(model.encode(comment, convert_to_tensor = False))
tmpnor = normalize(tmp, norm='l2')
tree = cKDTree(tmpnor)
# tree = cKDTree(tmp)
raw = pickle.dumps(tree)
with open("data.pkl", "wb") as f:
    pickle.dump(raw, f)