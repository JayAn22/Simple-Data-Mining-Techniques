import numpy as np
from collections import Counter
import re
import math

def convertString(s):
    
    return ''.join(s).lower().split()

def vectorize(text):
    word = re.compile(r'\w+')
    words = word.findall(text)
    return Counter(words)

def binaryJacard(n, m):
    intersection = np.logical_and(n, m)
    union = np.logical_or(n, m)
    similarity = intersection.sum() / union.sum() * 1.0
    return similarity

def stringJacard(list1, list2):
    cList1 = convertString(list1)
    cList2 = convertString(list2)
    s1, s2 = set(cList1), set(cList2)
    return len(s1 & s2) / len(s1 | s2) * 1.0

def cosineSim(list1, list2):
    text1 = list1
    text2 = list2
    sVec1 = vectorize(text1)
    sVec2 = vectorize(text2)
    intersection = set(sVec1.keys()) & set(sVec2.keys())
    #print(intersection)
    numerator = sum([sVec1[x] * sVec2[x] for x in intersection])    
    sumA = sum([sVec1[x] ** 2 for x in sVec1.keys()])
    sumB = sum([sVec2[x] ** 2 for x in sVec2.keys()])
    denominator = math.sqrt(sumA) * math.sqrt(sumB)    
    return numerator / denominator * 1.0

def bagOfWords(wordset, string):
    wordDict = dict.fromkeys(wordset,0)
    for word in string:
        wordDict[word]=string.count(word)
        
    v = []
    for i in wordDict:
        v.append(wordDict.get(i))    
    return v
        
if __name__ == "__main__":
    x = [0, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    y = [0, 1, 0, 0, 0, 1, 1, 0, 0, 0]
        
    sentenceA = "Data mining is the process of extracting and discovering patterns in large data sets involving methods at the intersection of machine learning, statistics, and database systems."
    sentenceB = "Data mining is the process of analyzing data, extracting patterns, discovering trends, and gaining insights using machine learning, database, statistics, and mathematics."
    sentenceC = "Coal mining is the process of extracting coal from the ground, involving discovering, exploration, and production."

    sentenceA2 = re.sub(r"[^a-zA-Z0-9]", " ", sentenceA.lower()).split()
    sentenceB2 = re.sub(r"[^a-zA-Z0-9]", " ", sentenceB.lower()).split()
    sentenceC2 = re.sub(r"[^a-zA-Z0-9]", " ", sentenceC.lower()).split()

    wordsetAB = np.union1d(sentenceA2, sentenceB2)
    bowAB = bagOfWords(wordsetAB, sentenceA2)
    bowAB2 = bagOfWords(wordsetAB, sentenceB2)
    
    wordsetAC = np.union1d(sentenceA2, sentenceC2)
    bowAC = bagOfWords(wordsetAC, sentenceA2)
    bowAC2 = bagOfWords(wordsetAC, sentenceC2)
    
    wordsetBC = np.union1d(sentenceB2, sentenceC2)
    bowBC = bagOfWords(wordsetBC, sentenceB2)
    bowBC2 = bagOfWords(wordsetBC, sentenceC2)
     
    print("{:=^40}".format("JACARD SIMILARITY"))
    print('X + Y' , binaryJacard(x, y)) 
    print('A + B' , stringJacard(sentenceA, sentenceB))    
    print('A + C' , stringJacard(sentenceA, sentenceC))
    print('B + C' , stringJacard(sentenceB, sentenceC))
    
    print()
    
    print("{:=^40}".format("COSINE SIMILARITY"))
    print('X + Y' , cosineSim(str(x),str(y))) 
    print('A + B' , cosineSim(sentenceA.lower(), sentenceB.lower()))    
    print('A + C' , cosineSim(sentenceA.lower(), sentenceC.lower()))
    print('B + C' , cosineSim(sentenceB.lower(), sentenceC.lower()))
    
    print()

    print("{:=^40}".format("CORRELATION"))
    cb = np.corrcoef(x, y)
    cab = np.corrcoef(bowAB, bowAB2)
    cac = np.corrcoef(bowAC, bowAC2)
    cbc = np.corrcoef(bowBC, bowBC2)    
    print('X + Y', cb[1, 0])
    print('A + B' , cab[1, 0])    
    print('A + C' , cac[1, 0])
    print('B + C' , cbc[1, 0])
    
    print()

    print("{:=^40}".format("EUCLIDEAN"))
    a = np.array(x)
    b = np.array(y)
    
    ab = np.array(bowAB)
    ab2 = np.array(bowAB2)
    
    ac = np.array(bowAC)
    ac2 = np.array(bowAC2)
    
    bc = np.array(bowBC)
    bc2 = np.array(bowBC2)
    
    print('X + Y' , np.linalg.norm(a - b))
    print('A + B' , np.linalg.norm(ab - ab2))    
    print('A + C' , np.linalg.norm(ac - ac2))
    print('B + C' , np.linalg.norm(bc - bc2))