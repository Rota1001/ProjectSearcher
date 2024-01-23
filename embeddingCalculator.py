import utils.fileLoader as fileLoader
import functools
import pickle
import torch
from sentence_transformers import SentenceTransformer, util
from scipy.spatial import cKDTree
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
# from angle_emb import AnglE

Comments = fileLoader.getComments(input("Please input your project directory:"))
#FileLoader.Download(input("Please input your github url:"))
#Comments = FileLoader.GetComments('./getFile')
#model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
# angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda() 
# model = SentenceTransformer('BAAI/bge-large-en')
#model = SentenceTransformer('sentence-t5-large')
#model = SentenceTransformer('KnutJaegersberg/infoxlm-large-sentence-embeddings')
model = SentenceTransformer('all-roberta-large-v1')

#model = SentenceTransformer('pritamdeka/S-Scibert-snli-multinli-stsb')
# model = SentenceTransformer('llmrails/ember-v1')
# model = SentenceTransformer('all-mpnet-base-v2')
#model.load_state_dict(torch.load('model.pt'))

model.cuda()
model.eval()
tmp = []

with open("comments.pkl", "wb") as f:
    pickle.dump(Comments, f)
for position, comment in tqdm(Comments):
   # tmp.append(list(angle.encode(comment, to_numpy=True)[0]))
    tmp.append(model.encode(comment, convert_to_tensor = False))
tmpnor = normalize(tmp, norm='l2')
tree = cKDTree(tmpnor)
# tree = cKDTree(tmp)
raw = pickle.dumps(tree)
with open("data.pkl", "wb") as f:
    pickle.dump(raw, f)