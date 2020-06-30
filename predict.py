# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from src.evals import Prediction
from src.model import RCNNSentimentNet, LSTMSentimentNet
from src.FL import FocalLoss
from config import Config


CONFIG = Config()
TEST_PATH = CONFIG.test_path
EMBEDDING_PATH = CONFIG.embedding_path
BATCH_SIZE = CONFIG.batch_size
MODEL_PATH = CONFIG.model_path
UNSEEN_PATH = CONFIG.unseen_path
PATH = CONFIG.path

# Checking if GPU is available or not
is_cuda = torch.cuda.is_available()
if is_cuda: DEVICE = torch.device("cuda")
else: DEVICE = torch.device("cpu")
print(DEVICE)


def Loader(data_x, batch_size):
    # Data Loader
    data = TensorDataset(torch.from_numpy(data_x))
    batch_size = batch_size
    data_loader = DataLoader(data, batch_size=batch_size)
    return data_loader


if __name__ == "__main__":
    
    # LOADING .NPZ UNSEEN DATA WITH FORMAT
    # COLUMNS 0 Date, 1 Time, 2 Text, 3 Cleantext, 4 Embedded data 
    data = np.load(UNSEEN_PATH, allow_pickle=True)
    data_dt = data['arr_0']
    data_ti = data['arr_1']
    data_rw = data['arr_2']
    data_txt = data['arr_3']
    data_x = data['arr_4']
    
    # LOADING EMBEDDING MATRIX
    embed = np.load(EMBEDDING_PATH, allow_pickle=True)
    embedding_matrix = embed['arr_0']
    
    # DATA LOADER
    data_loader = Loader(data_x, BATCH_SIZE)
    
    # MODEL PARAMS
    vocab_size = embedding_matrix.shape[0]
    embedding_dim = embedding_matrix.shape[1]
    model = RCNNSentimentNet(CONFIG, vocab_size, torch.Tensor(embedding_matrix))
    model.to(DEVICE)
    criterion = FocalLoss(reduction='mean')
    
    # LOADING MODEL STATE_DICT
    model.load_state_dict(torch.load(MODEL_PATH))

    # MAKING PREDICTION ON DATA
    predictions = Prediction(model, data_loader, criterion)
    
    # CREATING A CSV FILE
    date = pd.DataFrame(data_dt.tolist(), columns=['Date']) 
    time = pd.DataFrame(data_ti.tolist(), columns=['Time'])
    tweet = pd.DataFrame(data_rw.tolist(), columns=['Raw tweet'])
    txt = pd.DataFrame(data_txt.tolist(), columns=['Cleaned tweet'])
    pred = pd.DataFrame(predictions, columns=['Classes'])
    data = pd.concat([date, time, tweet, txt, pred], axis=1)
    # labels 0:sadness, 1:anger, 2:confident, 3:analytical, 4:fear
    emotions = ['Sadness' if cls==0 else 'Anger' if cls==1 else 'Confident' if cls ==2 else 'Analytical' if cls==3 else'Fear' for cls in data.Classes] 
    data.insert(5, 'emotion', emotions)
    data.head(10)
    data.to_csv(PATH+'/unseen data prediction.csv')
        
    
    
    