import sys
import src.FileLoader
import functools
import pickle
import torch
from sentence_transformers import SentenceTransformer, util
from scipy.spatial import cKDTree
import numpy as np
from tqdm import tqdm
from llama_cpp import Llama


model = "llama-pro-8b.Q2_K.gguf"
model_path = f"./model/{model}"
llm = Llama(
    model_path,
    n_ctx=2048,
    n_gpu_layers=-1,
    main_gpu=0,
    n_threads=1,
    embedding=True,
    verbose=False
)

Comments = src.FileLoader.GetComments(input("Please input your project directory:"))
tmp = []
with open("comments.pkl", "wb") as f:
    pickle.dump(Comments, f)
for position, comment in tqdm(Comments):
    #print(position)
   print(comment)
   a = llm.embed(comment)
   print(a)
   #tmp.append(llm.embed(comment))
tree = cKDTree(tmp)

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

