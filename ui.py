from customtkinter import *

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
from utils.normalizor import normalizor
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
model.cuda()

app = CTk()
app.geometry("720x640")
tabview : CTkTabview
#Tab of loading file
entry : CTkEntry
label : CTkLabel
loadFileBtn : CTkButton
loadWeights : CTkButton

#Tab of searching
userInput : CTkEntry
outputMessage : CTkTextbox
searchBtn : CTkButton

tree : cKDTree
comments = []

def deleteMessage():
    outputMessage.configure(state="normal")
    outputMessage.delete(0.0, 'end')
    outputMessage.configure(state="disabled")

def insertMessage(msg):
    outputMessage.configure(state="normal")
    outputMessage.insert('end', msg)
    outputMessage.configure(state="disabled")


def loadWeight(msg="Weight loaded"):
    global tree
    global comments
    try:
        with open("data.pkl", "rb") as f:
            raw = pickle.load(f)
            tree = pickle.loads(raw)
        with open("comments.pkl", "rb") as f:
            comments = pickle.load(f)
        if msg != '':
            label.configure(text=msg)
    except:
        label.configure(text="Fail to load weights")
            

def loadFile():
    global entry
    filePath = entry.get()
    comments = []
    global label
    try: 
        comments = fileLoader.getComments(filePath)
        tmp = []
        with open("comments.pkl", "wb") as f:
            pickle.dump(comments, f)
        for position, comment in tqdm(comments):
            tmp.append(model.encode(comment, convert_to_tensor=False))
        tmpnor = normalize(tmp, norm='l2')
        tree = cKDTree(tmpnor)
        raw = pickle.dumps(tree)
        with open("data.pkl", "wb") as f:
            pickle.dump(raw, f)
        label.configure(text="File loaded\r\nPath:"+filePath)
        loadWeight("")
    except:
        label.configure(text="Can't open this folder")
    entry.delete(0, 'end')


def search():
    global tree
    global comments
    if len(comments) == 0:
        deleteMessage()
        insertMessage("Weights not loaded yet")
        return
    if userInput.get() == "":
        deleteMessage()
        insertMessage("Please input something")
        return
    try:
        embedding = normalizor(model.encode(userInput.get(), convert_to_tensor=False))
        x, y = tree.query(embedding)
        position, comment = comments[y]
        deleteMessage()
        insertMessage(position)
        insertMessage("\r\n")
        insertMessage(comment)
    except:
        deleteMessage()
        insertMessage("Something Wrong")
def init():
    global tabview
    global entry
    global label
    global loadFileBtn
    global loadWeights
    tabview = CTkTabview(master=app)
    tabview.configure(width=600)
    tabview.configure(height=500)
    tabview.configure(CTkFont("Comic Sans MS", 20, "bold"))
    tabview.pack(padx=20, pady=20)
    tabview.add("Load File")
    tabview.add("Search")
    entry = CTkEntry(master=tabview.tab("Load File"), placeholder_text="Input your project path", font=("Comic Sans MS", 15, "bold"))
    entry.configure(width=400)
    label = CTkLabel(master=tabview.tab("Load File"), text="Hello", font=("Comic Sans MS", 15, "bold"))
    loadFileBtn = CTkButton(master=tabview.tab("Load File"), 
                        text="Load Files", 
                        corner_radius=32, 
                        fg_color="transparent", 
                        hover_color="#4158D0", 
                        border_color="#FFCC70", 
                        border_width=2, 
                        font=("Comic Sans MS", 15, "bold"),
                        command=loadFile
                        )
    loadWeights = CTkButton(master=tabview.tab("Load File"), 
                        text="Load Weights", 
                        corner_radius=32, 
                        fg_color="transparent", 
                        hover_color="#4158D0", 
                        border_color="#FFCC70", 
                        border_width=2, 
                        font=("Comic Sans MS", 15, "bold"),
                        command=loadWeight
                        )
    label.place(relx=0.5, rely=0.35,anchor="center")
    entry.place(relx=0.5, rely=0.5,anchor="center")
    loadFileBtn.configure(height=30)
    loadFileBtn.place(relx=0.35, rely=0.7, anchor="center")
    loadWeights.configure(height=30)
    loadWeights.place(relx=0.65, rely=0.7, anchor="center")
    global outputMessage
    global searchBtn
    global userInput
    outputMessage = CTkTextbox(master=tabview.tab("Search"), font=("Comic Sans MS", 15), state="disabled")
    searchBtn = CTkButton(master=tabview.tab("Search"), 
                        text="Search", 
                        corner_radius=32, 
                        fg_color="transparent", 
                        hover_color="#4158D0", 
                        border_color="#FFCC70", 
                        border_width=2, 
                        font=("Comic Sans MS", 15, "bold"),
                        command=search
                        )
    userInput = CTkEntry(master=tabview.tab("Search"), placeholder_text="Describe your request", font=("Comic Sans MS", 15, "bold"))
    userInput.configure(width=400)
    outputMessage.place(relx=0.5, rely=0.35, anchor="center")
    outputMessage.configure(width=500, height=300)
    userInput.place(relx=0.5, rely=0.8, anchor="center")
    searchBtn.place(relx=0.5, rely=0.9, anchor="center")

    



if __name__ == '__main__':
    init()
    app.mainloop()