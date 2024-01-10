import os
import re
from tqdm import tqdm

def Download(gitURL):
	os.system('rm -rf getFile')
	os.system('git clone ' + gitURL + ' getFile')

def LoadFiles(directory):
	files = []
	for file in os.listdir(directory):
		if os.path.isfile(os.path.join(directory, file)):
			root, extension = os.path.splitext(file)
			if extension == '.c':
				files.append(directory + '/' + file)
		else:
			temp = LoadFiles(directory + '/' + file)
			for x in temp:
				files.append(x)
	return files

def GetComments(directory):
	files = LoadFiles(directory)
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
					comment += '\n' + b
				finalList.append((address, comment))
				#cnt = 0
				# s = ""
				# for line in fp:
				# 	cnt += 1
				# 	for i in range(len(line) - 1):
				# 		if line[i] == '/' and line[i + 1] == '/':
				# 			comment = ''
				# 			for j in range(i + 2, len(line)):
				# 				comment += line[j]
				# 			s += " " + comment
				# 			#finalList.append((address[10:] + ', line: ' + str(cnt), comment))
				# 			break
				# finalList.append((address[10:], s))
		except:
			err += "Can't read " + address + "\n"
	print(err)
	print("Files Loaded")
	return finalList

def RemoveFile():
	os.system('rm -rf getFile')
