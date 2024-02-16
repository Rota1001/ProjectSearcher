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
import re
import threading
import os
import json

model = SentenceTransformer('all-roberta-large-v1')
if torch.cuda.is_available():
    model.cuda()

if not os.path.isdir("data"):
    os.mkdir("data")

app = CTk()
app.geometry("720x600")
app.focus()
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
k = 5
onlyShowImportant = BooleanVar(value=False)
pathLen: int
# cleanPathVar = BooleanVar(value=True)

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
    global label
    try:
        with open("data/data.pkl", "rb") as f:
            raw = pickle.load(f)
            tree = pickle.loads(raw)
        with open("data/comments.pkl", "rb") as f:
            comments = pickle.load(f)
        with open("data/information.json", "rb") as f:
            tmp = json.load(f)
            global pathLen
            pathLen = tmp["pathLen"]
        if msg != '':
            label.configure(text=msg)
    except:
        label.configure(text="Fail to load weights")
            
def cleanPathOption():
   # onlyShowImportant = check_var.get()
    print("hello")

def loadFile():
    global entry
    filePath = entry.get()
    if filePath == "":
        filePath = filedialog.askdirectory()
    global pathLen
    pathLen = len(filePath)
    with open("data/information.json", "w") as f:
            tmp = {}
            tmp["pathLen"] = pathLen
            json.dump(tmp, f)
    comments = []
    global label
    try: 
        entry.delete(0, 'end')
        entry.insert(0, filePath)
        label.configure(text="File Loading...")
        comments = fileLoader.getComments(filePath)
        tmp = []
        with open("data/comments.pkl", "wb") as f:
            pickle.dump(comments, f)
        commentLen = len(comments)
        cnt = 0
        label.configure(text="Start Embedding")
        for position, comment in tqdm(comments):
            cnt += 1
            x = position[pathLen + 1:]
            # x = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])","",x)
            tmp.append(model.encode(x, convert_to_tensor=False))
            if cnt % 10 == 0:
                label.configure(text="Embedding: {:.2f}%".format(cnt / commentLen * 100))
        tmpnor = normalize(tmp, norm='l2')
        tree = cKDTree(tmpnor)
        raw = pickle.dumps(tree)
        with open("data/data.pkl", "wb") as f:
            pickle.dump(raw, f)
        label.configure(text="File loaded\r\nPath:"+filePath)
        loadWeight("")
    except:
        label.configure(text="Can't open this folder")
    entry.delete(0, 'end')

def loadTheFile():
    loading = threading.Thread(target=loadFile)
    loading.start()

def search():
    global tree
    global comments
    global k
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
        y = tree.query(embedding, k)[1]
        deleteMessage()
        for i in y:
            position, comment = comments[i]
            if onlyShowImportant.get():
                position = position[pathLen:]
            insertMessage(position)
            insertMessage("\r\n")
       # insertMessage(comment)
    except:
        deleteMessage()
        insertMessage("Something Wrong")

def modelSelect(value):
    global comments
    global model
    global label
    model = SentenceTransformer(value)
    if torch.cuda.is_available():
        model.cuda()
    comments = []
    print("done")
    label.configure(text="")

def kSelect(value):
    global k
    k = int(value)

def loadFileInit():
    global tabview
    global entry
    global label
    global loadFileBtn
    global loadWeights
    tabview.add("  Load File  ")
    entry = CTkEntry(master=tabview.tab("  Load File  "), placeholder_text="Input your project path", font=("Comic Sans MS", 15, "bold"))
    entry.configure(width=400)
    label = CTkLabel(master=tabview.tab("  Load File  "), text="Hello", font=("Comic Sans MS", 15, "bold"))
    loadFileBtn = CTkButton(master=tabview.tab("  Load File  "), 
                        text="Load Files", 
                        corner_radius=32, 
                        fg_color="transparent", 
                        hover_color="#4158D0", 
                        border_color="#FFCC70", 
                        border_width=2, 
                        font=("Comic Sans MS", 15, "bold"),
                        command=loadTheFile
                        )
    loadWeights = CTkButton(master=tabview.tab("  Load File  "), 
                        text="Load Weights", 
                        corner_radius=32, 
                        fg_color="transparent", 
                        hover_color="#4158D0", 
                        border_color="#FFCC70", 
                        border_width=2, 
                        font=("Comic Sans MS", 15, "bold"),
                        command=loadWeight
                        )
    label.place(relx=0.5, rely=0.25,anchor="center")
    entry.place(relx=0.5, rely=0.4,anchor="center")
    loadFileBtn.configure(height=30)
    loadFileBtn.place(relx=0.35, rely=0.6, anchor="center")
    loadWeights.configure(height=30)
    loadWeights.place(relx=0.65, rely=0.6, anchor="center")

def searchInit():
    global tabview
    tabview.add("  Search  ")
    global outputMessage
    global searchBtn
    global userInput
    outputMessage = CTkTextbox(master=tabview.tab("  Search  "), font=("Comic Sans MS", 15), state="disabled", corner_radius=20)
    searchBtn = CTkButton(master=tabview.tab("  Search  "), 
                        text="Search", 
                        corner_radius=32, 
                        fg_color="transparent", 
                        hover_color="#4158D0", 
                        border_color="#FFCC70", 
                        border_width=2, 
                        font=("Comic Sans MS", 15, "bold"),
                        command=search
                        )
    userInput = CTkEntry(master=tabview.tab("  Search  "), placeholder_text="Describe your request", font=("Comic Sans MS", 15, "bold"))
    userInput.configure(width=400)
    outputMessage.place(relx=0.5, rely=0.35, anchor="center")
    outputMessage.configure(width=500, height=300)
    userInput.place(relx=0.5, rely=0.8, anchor="center")
    searchBtn.place(relx=0.5, rely=0.9, anchor="center")

def tabInit():
    global tabview
    tabview = CTkTabview(master=app, corner_radius=20)
    tabview.configure(width=600)
    tabview.configure(height=500)
    tabview.configure(CTkFont("Comic Sans MS", 20, "bold"))
    tabview.pack(padx=20, pady=20)
    
def modelSettingInit():
    global tabview
    tabview.add("Model Setting")
    modelCombobox = CTkComboBox(master=tabview.tab("Model Setting"), values=["all-roberta-large-v1", "paraphrase-multilingual-MiniLM-L12-v2"], font=("Comic Sans MS", 15, "bold"), command=modelSelect)
    modelCombobox.configure(width=400)
    modelCombobox.place(relx=0.55, rely=0.3, anchor="center")
    modelLabel = CTkLabel(master=tabview.tab("Model Setting"), text="Model", font=("Comic Sans MS", 15, "bold"))
    modelLabel.place(relx=0.13, rely = 0.3, anchor="center")

    kCombobox =  CTkComboBox(master=tabview.tab("Model Setting"), values=["1", "2", "5", "10", "20", "100"], font=("Comic Sans MS", 15, "bold"), command=kSelect)
    kCombobox.set("5")
    kCombobox.place(relx=0.55, rely=0.5, anchor="center")
    kCombobox.configure(width=400)
    kLabel = CTkLabel(master=tabview.tab("Model Setting"), text="k", font=("Comic Sans MS", 15, "bold"))
    kLabel.place(relx=0.13, rely = 0.5, anchor="center")

    checkBox = CTkCheckBox(master=tabview.tab("Model Setting"), text="Show clean path", font=("Comic Sans MS", 15, "bold"),variable=onlyShowImportant, onvalue=True, offvalue=False)
    checkBox.place(relx=0.25, rely=0.7, anchor="center")

def init():
    global app
    app.title("Project Searcher")
    set_appearance_mode("dark")
    set_default_color_theme("dark-blue")
    tabInit()
    loadFileInit()
    modelSettingInit()
    searchInit()
    



if __name__ == '__main__':
    init()
    app.mainloop()
