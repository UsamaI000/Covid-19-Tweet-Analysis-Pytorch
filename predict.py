# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from src.evals import Prediction
from src.model import RCNNSentimentNet, LSTMSentimentNet
from config import Config


CONFIG = Config()
TEST_PATH = CONFIG.test_path
EMBEDDING_PATH = CONFIG.embedding_path
BATCH_SIZE = 1000
MODEL_PATH = CONFIG.model_path
UNSEEN_PATH = CONFIG.unseen_path
PATH = CONFIG.path
DEVICE = CONFIG.device

# Data Loader
def Loader(data_x, batch_size):
    data = TensorDataset(torch.from_numpy(data_x))
    batch_size = batch_size
    data_loader = DataLoader(data, batch_size=batch_size)
    return data_loader


if __name__ == "__main__":
    
    # LOADING .NPZ UNSEEN DATA WITH FORMAT
    # COLUMNS 0: Date, 1: Time, 2: Text, 3: place, 4: co-or 5: country 6: clean txt 7: vector 
    data = np.load(UNSEEN_PATH, allow_pickle=True)
    data_dt = data['arr_0']
    data_ti = data['arr_1']
    data_rw = data['arr_2']
    data_plc = data['arr_3']
    data_coor = data['arr_4']
    data_cntry = data['arr_5']
    data_txt = data['arr_6']
    data_x = data['arr_7']
    
    # LOADING EMBEDDING MATRIX
    embed = np.load(EMBEDDING_PATH, allow_pickle=True)
    embedding_matrix = embed['arr_0']
    
    # DATA LOADER
    data_loader = Loader(data_x, BATCH_SIZE)
    
    # MODEL PARAMS
    vocab_size = embedding_matrix.shape[0]
    model = RCNNSentimentNet(CONFIG, vocab_size, torch.Tensor(embedding_matrix))
    model.to(DEVICE)

    # LOADING MODEL STATE_DICT
    model.load_state_dict(torch.load(MODEL_PATH))

    # MAKING PREDICTION ON DATA
    predictions = Prediction(model, data_loader)
    
    # CREATING A CSV FILE
    date = pd.DataFrame(data_dt.tolist(), columns=['date']) 
    time = pd.DataFrame(data_ti.tolist(), columns=['time'])
    tweet = pd.DataFrame(data_rw.tolist(), columns=['text'])
    place = pd.DataFrame(data_plc.tolist(), columns=['place'])
    cor = pd.DataFrame(data_coor.tolist(), columns=['coordinates'])
    country = pd.DataFrame(data_cntry.tolist(), columns=['country'])
    txt = pd.DataFrame(data_txt.tolist(), columns=['clean_txt'])
    pred = pd.DataFrame(predictions, columns=['labels'])
    data = pd.concat([date, time, tweet, place, cor, country, txt, pred], axis=1)
    # labels 0:sadness, 1:anger, 2:confident, 3:analytical, 4:fear
    emotions = ['Sadness' if cls==0 else 'Fear' if cls==1 else 'Confident' if cls ==2 else 'Analytical' if cls==3 else'Anger' for cls in data.labels] 
    data.insert(8, 'emotions', emotions)
    data.to_csv(PATH+'/Predictions.csv', index=False)
        
    
    
    