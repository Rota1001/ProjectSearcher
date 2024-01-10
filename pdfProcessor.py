import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch

def extract_titles_and_paragraphs(pdf_path):
    doc = fitz.open(pdf_path)
    
    title_paragraphs = []
    current_title = ""
    current_paragraph = ""
    lasttitle = ""
    now_paragraphs = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        dict = page.get_text("dict")
        blocks = dict["blocks"]
        for block in blocks:
            if not "lines" in block.keys():
                continue
            for span in block["lines"]:
                data = span["spans"]
                for line in data:
                    # print(lasttitle)
                    if(len(line["text"]) < 4):
                        continue
                    if(line["size"] > 12):
                        if(line["text"][0:7] == "CHAPTER" or line["size"] > 20):
                            if(lasttitle != ""):
                                title_paragraphs.append((lasttitle, now_paragraphs))
                            lasttitle = ""
                            now_paragraphs = ""
                            continue
                        if lasttitle == "":
                            lasttitle = line["text"]
                            now_paragraphs = ""
                            continue
                        title_paragraphs.append((lasttitle, now_paragraphs))
                        now_paragraphs = ""
                        lasttitle = line["text"]
                    else:
                        now_paragraphs += " " + line["text"]
    
    doc.close()
    
    return title_paragraphs

# 要提取的 PDF 文件路徑
pdf_path = r"C:\Users\johnn\OneDrive\桌面\finaldocument.pdf"

# 提取標題和內文列表
titles_and_paragraphs = extract_titles_and_paragraphs(pdf_path)

train_examples = []

# 打印結果
for i, (title, paragraph) in enumerate(titles_and_paragraphs, start=1):
    #  print(title)
    #  print(paragraph)
     train_examples.append(InputExample(texts=[title, paragraph], label=0.9))
#     print(f"內文 {i}: {paragraph}")
#     print("=" * 50)
model = SentenceTransformer('all-mpnet-base-v2')
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)
torch.save(model.state_dict(), 'model.pt')
# model.load_state_dict(torch.load('model.pt'))