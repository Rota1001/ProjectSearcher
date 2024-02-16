import os
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

# model = SentenceTransformer('all-roberta-large-v1')
# model.cuda()


def download(gitURL):
	os.system('rm -rf getFile')
	os.system('git clone ' + gitURL + ' getFile')

def loadFiles(directory):
	files = []
	for file in os.listdir(directory):
		if os.path.isfile(os.path.join(directory, file)):
			root, extension = os.path.splitext(file)
			if extension == '.c':
				files.append(directory + '/' + file)
		else:
			temp = loadFiles(directory + '/' + file)
			for x in temp:
				files.append(x)
	return files

# forbidden = ["is", "am", "are", "This", "for", "the", "of", "on", "the", "a", "easy", "using", "into", "and", "had", "that", "That", "this", "The", "with", "one", "as", "soon", "possible", "in", "here", "depends", "on", "itself", "yourself", "himself", "herself", "Thus", "thus", "at", "return", "but", "we", "do", "in", "case", "The", "no", "it", "doesn't", "even", "Note", "because", "going", "A", "Might", "might", "rather", "than", "to", "seems", "seem", "be", "below", "cause", "won't", "where", "what", "when", "Where", "What", "When", "first", "second", "then", "its", "Copyright", "(C)", "1997", "Jay", "Estabrook", "very"]


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
				for word in a:
					comment += word
				# comment = re.sub(r"[^a-zA-Z]", "", comment)
			#	if len(comment) > 100:
				finalList.append((address, comment))
		except:
			err += "Can't read " + address + "\n"
	print(err)
	print("Files Loaded")
	return finalList

def RemoveFile():
	os.system('rm -rf getFile')
