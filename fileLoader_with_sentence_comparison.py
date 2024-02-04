import os
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-roberta-large-v1')
# model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
model.cuda()


def download(gitURL):
	os.system('rm -rf getFile')
	os.system('git clone ' + gitURL + ' getFile')

def loadFiles(directory):
	files = []
	for file in os.listdir(directory):
		if os.path.isfile(os.path.join(directory, file)):
			root, extension = os.path.splitext(file)
			if extension == '.c' or extension == '.h':
				files.append(directory + '/' + file)
		else:
			temp = loadFiles(directory + '/' + file)
			for x in temp:
				files.append(x)
	return files

# forbidden = ["is", "am", "are", "This", "for", "the", "of", "on", "the", "a", "easy", "using", "into", "and", "had", "that", "That", "this", "The", "with", "one", "as", "soon", "possible", "in", "here", "depends", "on", "itself", "yourself", "himself", "herself", "Thus", "thus", "at", "return", "but", "we", "do", "in", "case", "The", "no", "it", "doesn't", "even", "Note", "because", "going", "A", "Might", "might", "rather", "than", "to", "seems", "seem", "be", "below", "cause", "won't", "where", "what", "when", "Where", "What", "When", "first", "second", "then", "its", "Copyright", "(C)", "1997", "Jay", "Estabrook", "very"]

def score(x, y):
	embeddings1 = model.encode(x, convert_to_tensor=True)
	return util.cos_sim(embeddings1, y)


def getComments(directory):
	files = loadFiles(directory)
	finalList = []
	err = ""
	for address in tqdm(files):
		try:
			with open(address, 'r', encoding= 'utf-8') as fp:
				s = ""
				for line in fp:
					s += line
				a = re.findall(r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"', s,re.DOTALL | re.MULTILINE)
				comment = ""

				# fileName = address[:-1]
				# fileName = model.encode(fileName, convert_to_tensor=True)
				# words = a


				fileName = re.sub(r"[^a-zA-Z]", "", address[39:-1])
				fileName = model.encode(fileName, convert_to_tensor=True)
				words = [re.sub(r"[^a-zA-Z]", "", b) for b in a]
				embeddings = model.encode(words, convert_to_tensor=True)
				similarity = util.cos_sim(embeddings, fileName).cpu().numpy()


				for i in range(len(similarity)):
					if similarity[i][0] > 0.6:
						comment += words[i]				



				# for b in a:
				# 	b = re.sub(r"[^a-zA-Z]", "", b)
				# 	if score(b, fileName) > 0.7:
				# 		comment += b
				# cmt = comment.split(' ')
				# comment = ""
				# for word in cmt:
				# 	if word not in forbidden:
				# 		comment += word
				# comment = re.sub(r"[^a-zA-Z]", "", comment)
				#comment = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])","",comment)
				if len(comment) > 0:
					finalList.append((address, comment))
		except:
			err += "Can't read " + address + "\n"
	print(err)
	print("Files Loaded")
	return finalList

def RemoveFile():
	os.system('rm -rf getFile')
