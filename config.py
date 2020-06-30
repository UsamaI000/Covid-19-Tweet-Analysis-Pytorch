# -*- coding: utf-8 -*-
# config.py
import nltk
from nltk.corpus import stopwords
from nltk import SnowballStemmer

class Config(object):    
    # TEXT CLENAING
    TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
    
    # Stop words
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    stemmer = SnowballStemmer('english')
    
    # Model params
    batch_size = 100
    embed_size = 300
    hidden_layers = 2
    hidden_size = 128  #64
    output_size = 5
    hidden_size_linear = 128 #256 #512 #128
    dropout_keep = 0.51
    lr=5e-1
    epochs = 300
    
    # Directories path
    model_path = "../drive/My Drive/results/fl/state_dict.pt"
    embedding_path = "../drive/My Drive/embedding_matrix_np.npz"
    train_path = "../drive/My Drive/trainset_np.npz"
    test_path = "../drive/My Drive/testset_np.npz"
    tokenizer_path = "../drive/My Drive/tokenizer.pickle"
    path = "../results/fl"
    unseen_path = "../drive/My Drive/unseen_data_np.npz"
