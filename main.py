import sys
sys.path.append(r'src')
import FileLoader
import Scorer
import Compare
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
#model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
# model = SentenceTransformer('pritamdeka/S-Scibert-snli-multinli-stsb')
model = SentenceTransformer('all-mpnet-base-v2')
model.load_state_dict(torch.load('model.pt'))
Comments = FileLoader.GetComments(input("Please input your project directory:"))
corpus = []
model.cuda()
model.eval()
with open("comments.pkl", "wb") as f:
    pickle.dump(Comments, f)
for position, comment in tqdm(Comments):
    #print(position)
    #print(comment)
    corpus.append(comment)
#    tmp.append(model.encode(comment, convert_to_tensor = False))
tmp = model.encode(corpus, show_progress_bar=True, convert_to_tensor=False)
tmpnor = normalize(tmp, norm='l2')
tree = cKDTree(tmpnor)
# tree = cKDTree(tmp)
raw = pickle.dumps(tree)
with open("data.pkl", "wb") as f:
    pickle.dump(raw, f)



#model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
#targets = ['yeeeee', 'yeeeeeee', 'yeeeeeeah']
#FinalList = []
#for position, comment in Comments:
#	score = Scorer.Score(comment, targets)
#	FinalList.append((-score, position))

#FinalList.sort()
#print("========================================")
#print("Score\t|\tPosition")
#print("========================================")
#for score, position in FinalList:
#	if -score > 70: 
#		print(round(-score, 2), end = '%\t|\t')
#		print(position)

