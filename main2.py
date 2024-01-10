
import torch
from sentence_transformers import SentenceTransformer, util
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import pickle

model = SentenceTransformer('all-mpnet-base-v2')
model.load_state_dict(torch.load('model.pt'))
model.eval()
# model = SentenceTransformer('pritamdeka/S-Scibert-snli-multinli-stsb')
#model = SentenceTransformer('bert-base-uncased')
tree : cKDTree
Comments = []

with open("data.pkl", "rb") as f:
    raw = pickle.load(f)
    tree = pickle.loads(raw)

with open("comments.pkl", "rb") as f:
    Comments = pickle.load(f)

while True:
    sentence = input("Input sentence:")
    embedding = model.encode(sentence, convert_to_tensor = False)
    arr = []
    arr.append(embedding)
    embedding = normalize(arr, norm='l2')[0]
    print("==================================================================")
    # print(embedding)
    x, y = tree.query(embedding)
    position, comment = Comments[y]
    print(position)
    print(comment)
    print("==================================================================")
    #for b in y:
        # position, comment = Comments[b]
        # print(position)
        # print(comment)
        # print("==================================================================")