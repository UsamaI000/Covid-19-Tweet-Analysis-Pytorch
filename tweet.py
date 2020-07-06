import nltk
import re
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.nn import functional as F
from src.model import RCNNSentimentNet
from keras.preprocessing.sequence import pad_sequences
from config import Config
from src.utils import preprocess
from torch.utils.data import TensorDataset, DataLoader


CONFIG = Config()
TOKENIZER =  CONFIG.tokenizer_path
RE = CONFIG.TEXT_CLEANING_RE
EMBEDDING_PATH = CONFIG.embedding_path
MODEL_PATH = CONFIG.model_path
DEVICE = CONFIG.device
print(DEVICE)

def Loader(data_x):
    # Data Loader
    data = TensorDataset(torch.from_numpy(data_x))
    batch_size = 1
    data_loader = DataLoader(data, batch_size=batch_size)
    return data_loader


if __name__ == "__main__":

    # LOADING EMBEDDING MATRIX
    embed = np.load(EMBEDDING_PATH, allow_pickle=True)
    embedding_matrix = embed['arr_0']  

    # INPUTTING THE TWEET
    tweet = input("\nInput a tweet: \n")
    with open(TOKENIZER, 'rb') as handle:
        tokenizer = pickle.load(handle)
    sentence = [preprocess(tweet)]

    print('\nCleaned tweet is:\n', sentence[0]+'\n')
    
    # TOKENIZATION
    vector = pad_sequences(tokenizer.texts_to_sequences(sentence), maxlen=300)

    # DATA LOADER
    data_loader = Loader(vector)

    # MODEL PARAMS
    vocab_size = embedding_matrix.shape[0]
    model = RCNNSentimentNet(CONFIG, vocab_size, torch.Tensor(embedding_matrix))
    model.to(DEVICE)

    # LOADING MODEL STATE_DICT
    model.load_state_dict(torch.load(MODEL_PATH))

    model.eval()
    for inputs in data_loader:
        inputs = inputs[0].to(DEVICE)
        outputs, _ = model.forward(inputs.type(torch.long))
        outputs = F.softmax(outputs, dim=1)
        _, pred = torch.max(outputs, dim=1)
        predicted = pred.cpu().numpy()

    emots = ['Sadness', 'Fear', 'Confident', 'Analytical', 'Anger']
    scores = [emots[i]+': '+str((outputs[0][i]).detach().cpu().numpy()) for i in range(0,5)]
    for each in scores: print('The predicted score for', each)

    for i in range(len(emots)):
        if predicted==i: emotion=emots[i]
    print('\nThe predicted emotion for tweet is:',emotion)



