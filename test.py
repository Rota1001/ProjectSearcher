import re
path = r"C:\Users\johnn\OneDrive\桌面\linux-3.1.1\kernel\acct.c"
s = ""
with open(path, "r", encoding="utf-8") as fp:
     for line in fp:
         s += line + "\n"

def findFunc(s):
    s = re.sub("#.*", "", s)
    d = re.findall(r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"', s,re.DOTALL | re.MULTILINE)
    for i in d:
        s = s.replace(i, '')
    s = re.sub('.*;', "", s)
    s = re.sub("if(.*)", "", s)
    s = re.sub("for(.*)", "", s)
    s = re.sub("while(.*)", "", s)
    k = re.findall(".*\(.*\)", s)
    ret = ""
    for i in k:
        i = re.sub("\(.*\)", "", i)
        ret += i.split(' ')[-1]
    ret = re.sub(r"[^a-zA-Z]", "", ret)
    return ret

print(findFunc(s))

