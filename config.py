# -*- coding: utf-8 -*-
# config.py
import nltk
import torch
from nltk.corpus import stopwords
from nltk import SnowballStemmer

class Config(object):    
    # TEXT CLENAING
    TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
    
    # Checking if GPU is available or not
    is_cuda = torch.cuda.is_available()
    if is_cuda: device = torch.device("cuda")
    else: device = torch.device("cpu")

    # Stop words
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    stemmer = SnowballStemmer('english')
    
    # Model params
    batch_size = 64            # Batch size 
    embed_size = 300           # Word2Vec Embedding size       
    hidden_layers = 3          # Number of Hidden layers for Bi-directional LSTM
    hidden_size = 100          # Size of each Hidden layer in LSTM
    output_size = 5            # Output size 
    hidden_size_linear = 724   # Fully connected layers
    hidden_size_linear2 = 512  # Fully connected layers
    hidden_size_linear3 = 256  # Fully connected layers
    dropout_keep = 0.41        # Dropout layer probability
    lr= 0.005                   # Learning rate
    epochs = 500               # Number of Epochs
    
    # Directories path
    model_path = "../drive/My Drive/Final/state_dict.pt"
    embedding_path = "../drive/My Drive/Final/embedding_matrix_np.npz"
    train_path = ""
    test_path = ""
    tokenizer_path = "../drive/My Drive/Final/w2v_tokenizer.pickle"
    path = "../drive/My Drive/Final"
    unseen_path = "../drive/My Drive/Final/unseen_data_np.npz"
