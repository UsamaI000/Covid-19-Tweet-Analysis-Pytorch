# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.nn import functional as F
from config import Config
from sklearn.metrics import f1_score
from src.utils import Confusion_matrix, plotCurves, Dataloader
from src.evals import Validation_eval
from src.model import RCNNSentimentNet, LSTMSentimentNet
from src.losses import FocalLoss, CB_Loss

CONFIG = Config()

# Configurations for the training setup
BATCH_SIZE = CONFIG.batch_size
TRAIN_PATH = CONFIG.train_path
TEST_PATH = CONFIG.test_path
EMBEDDING_PATH = CONFIG.embedding_path
LR = CONFIG.lr
EPOCHS = CONFIG.epochs
PATH = CONFIG.path
MODEL_PATH = CONFIG.model_path
DEVICE = CONFIG.device
print('Started training using', DEVICE)

# Training function
def run(model, epochs, criterion, optimizer, train_loader, val_loader, device, path, load_model=False):
    clip = 5
    valid_loss_min = np.Inf
    stats = []
    if load_model: model.load_state_dict(torch.load(MODEL_PATH))
    print('\nTraining Started\n')
    for epoch in range(0, epochs):
        train_acc = 0.0
        val_acc = 0.0  
        train_losses = []
        train_pred, train_orig, valid_pred, valid_orig = [], [], [], []
        model.train()
        print('Epoch: {}/{}...'.format(epoch+1, epochs))
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            model.zero_grad()
            output, _ = model.forward(inputs.type(torch.long))
            loss = criterion(output, labels.squeeze(1).long())
            train_losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
    
            # Calculate train accuracy
            output = F.softmax(output, dim=1)
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(labels.data.view_as(pred))
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            train_acc += accuracy.item() * inputs.size(0)
            train_pred.extend(pred.cpu().numpy())
            train_orig.extend(labels.cpu().numpy())
        train_acc = train_acc / len(train_loader.dataset)
        print("\tTrain Loss: {:.6f}".format(loss.item())+", Train accuracy: {:.3f}%".format(100*train_acc))
        train_f1 = f1_score(np.array(train_orig), np.array(train_pred), average='micro')
        print(f'\tTrain F1 Score: {train_f1:.6f}')
            
        # Running model on validation data
        with torch.no_grad():
            val_losses, val_acc, val_f1, vo, vp , valid_loss_min = Validation_eval(model, val_loader, criterion, val_acc, valid_orig, valid_pred, valid_loss_min)
            stats.append([np.mean(train_losses), np.mean(val_losses), train_acc, val_acc, train_f1, val_f1])
                 
        # Plotting confusion matrix after some epochs
        if epoch % 3 == 0: 
          Confusion_matrix(np.array(vo), np.array(vp), title ='Validation Confusion matrix', labels = ['sadness','fear','confident','analytical','anger'], path=path+'/validation cm.png')    
          Confusion_matrix(np.array(train_orig), np.array(train_pred), title ='Training Confusion matrix', labels = ['sadness','fear','confident','analytical','anger'], path=path+'/train cm.png')    

        stat = pd.DataFrame(stats, columns=['train_loss', 'valid_loss', 'train_acc','valid_acc', 'train_f1', 'val_f1'])
        if epoch % 5 == 0: plotCurves(stat, path, f1=True)
        if epoch % 9 == 0: torch.save(model.state_dict(), path+'/state_dict.pt')
    print('Finished Training')
    
    
if __name__ == "__main__":
    # LOADING TRAINING AND VALIDATION .NPZ FILES
    train_data = np.load(TRAIN_PATH, allow_pickle=True)
    test_data = np.load(TEST_PATH, allow_pickle=True)
    embed = np.load(EMBEDDING_PATH, allow_pickle=True)
    train_x = train_data['arr_4']
    train_y = train_data['arr_5']
    valid_x = test_data['arr_4']
    valid_y = test_data['arr_5']
    embedding_matrix = embed['arr_0']
    
    # CREATING DATA LOADER TO FEED TO THE MODEL
    train_loader = Dataloader(train_x, train_y, BATCH_SIZE)
    val_loader = Dataloader(valid_x, valid_y, BATCH_SIZE*2)
  
    # MODEL PARAMS
    vocab_size = embedding_matrix.shape[0]
    
    # MODEL INSTANTIATION
    model = RCNNSentimentNet(CONFIG, vocab_size, torch.Tensor(embedding_matrix)).to(DEVICE)
    
    # LOSS FUNCTION AND OPTIMIZATION FUNCTION
    weights = [1.50, 3.56, 1.58, 1.0, 2.55]
    class_weights = torch.FloatTensor(weights).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    #alpha = np.array([[0.767],[0.9055],[0.7872],[0.6632],[0.8681]])
    #alpha = torch.Tensor(alpha).type(torch.FloatTensor).to(DEVICE)    
    #criterion = FocalLoss(alpha=alpha)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    
    # START TRAINING
    run(model, EPOCHS, criterion, optimizer, train_loader, val_loader, DEVICE, PATH, load_model=False)