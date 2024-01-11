import os
import re
from tqdm import tqdm

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
				for b in a:
					comment += b
				comment = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])","",comment)
				if comment != "":
					finalList.append((address, comment))
		except:
			err += "Can't read " + address + "\n"
	print(err)
	print("Files Loaded")
	return finalList

def RemoveFile():
	os.system('rm -rf getFile')
