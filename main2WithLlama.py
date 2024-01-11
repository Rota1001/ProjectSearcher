
import torch
from sentence_transformers import SentenceTransformer, util
from scipy.spatial import cKDTree
import pickle
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

tree : cKDTree
Comments = []

with open("data.pkl", "rb") as f:
    raw = pickle.load(f)
    tree = pickle.loads(raw)

with open("comments.pkl", "rb") as f:
    Comments = pickle.load(f)

while True:
    sentence = input("Input sentence:")
    embedding = llm.embed(sentence)
    print("==================================================================")
    #a, b = tree.query(embedding)
    #print(Comments[b])
    x, y = tree.query(embedding)
    position, comment = Comments[y]
    print(position)
    print(comment)
    #for b in y:
        # position, comment = Comments[b]
        # print(position)
        # print(comment)
        # print("==================================================================")