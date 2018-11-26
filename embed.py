# from pkl import read
from pkl import read
import os
import numpy as np
def load_vocab():
    vocab_file = os.path.join(os.getcwd(),"vocab.pkl")
    pklData = read(vocab_file) # list ['我','是']
    vocab = {} 
    for idx, item in enumerate(pklData):
        vocab[item] = idx  # .strip()

    return vocab 

vocab = load_vocab()
embedding_file = os.path.join(os.getcwd(),"embeddings.pkl")
trainedEmbeddings = read(embedding_file) # 长度 4549 3个手工增加 3个预留
# print(embeddings[100])

def str2embed(string):
    embeddingList = []
    for word in string:
        embedding = trainedEmbeddings[vocab.get(word)]
        embedding = np.array(embedding)
        embeddingList.append(embedding)
    return np.array(embeddingList)

# str2embed('string')[1]

# def getSize():
#     return len(pklData)

# def getIndexFromChar(character):
#     try:
#         return pklData.index(character)
#     except:
#         return False

# def getCharFromIndex(index):
#     try:
#         return pklData[index]
#     except:
#         return False

