# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import f1_score
from config import Config
from src.utils import Confusion_matrix

CONFIG = Config()

is_cuda = torch.cuda.is_available()
if is_cuda: device = torch.device("cuda")
else: device = torch.device("cpu")
print(device)

# METHOD FOR TEST DATASET EVALUATION
def Test_Eval(model, data_loader, criterion):
  test_losses = []
  predict, orig = [], []
  acc=0
  model.eval()
  for inputs, labels in data_loader:
      inputs, labels = inputs.to(device), labels.to(device)
      output, _ = model.forward(inputs.type(torch.long))
      test_loss = criterion(output, labels.squeeze(1).long())
      test_losses.append(test_loss.item())
      
      # Calculate accuracy
      output = F.softmax(output, dim=1)
      _, pred = torch.max(output, dim=1)
      correct_tensor = pred.eq(labels.data.view_as(pred))
      accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
      acc += accuracy.item() * inputs.size(0)
      predict.extend(pred.cpu().numpy())
      orig.extend(labels.cpu().numpy())
  acc = acc / len(data_loader.dataset)
  f1 = f1_score(np.array(orig), np.array(predict), average='micro')
  Confusion_matrix(np.array(orig), np.array(predict), title ='Test Confusion matrix', labels = ['sadness','anger','confident','analytical','fear'], path='../test cm.png')
  return np.mean(test_losses), acc, f1


# METHOD FOR PREDICTING LABELS ON UNSEEN DATA
def Prediction(model, data_loader, criterion):
  predictions = []
  model.eval()
  for inputs in data_loader:
      inputs = inputs[0].to(device)
      output, _ = model.forward(inputs.type(torch.long))
      output = F.softmax(output, dim=1)
      _, pred = torch.max(output, dim=1)
      predictions.extend(pred.cpu().numpy())
  return predictions


# METHOD FOR EVALUTAING VALIDATION DATASET PERFORMANCE
def Validation_eval(model, val_loader, criterion, va, vp, vo, valid_loss_min):
    model.eval()
    val_losses = []
    for inp, target in val_loader:
        inp, target = inp.to(device), target.to(device)
        out, _ = model.forward(inp.type(torch.long))
        val_loss = criterion(out, target.squeeze(1).long())
        val_losses.append(val_loss.item())

        # Calculate validation accuracy
        output = F.softmax(out, dim=1)
        _, val_pred = torch.max(output, dim=1)
        val_correct_tensor = val_pred.eq(target.data.view_as(val_pred))
        accuracy = torch.mean(val_correct_tensor.type(torch.FloatTensor))
        va += accuracy.item() * inp.size(0)
        vp.extend(val_pred.cpu().numpy())
        vo.extend(target.cpu().numpy())
    val_acc = va / len(val_loader.dataset)
    print("\tValidation Loss: {:.6f}".format(val_loss)+", Validation accuracy: {:.3f}%".format(val_acc*100))
    val_f1 = f1_score(np.array(vo), np.array(vp), average='micro')
    print(f'\tValid F1 Score: {val_f1:.6f}')
        
    # Saving model if mean of val loss is smaller than min val loss
    if np.mean(val_losses) <= valid_loss_min:
        torch.save(model.state_dict(), CONFIG.path+'/state_dict.pt')
        print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...'.format(valid_loss_min,np.mean(val_losses))+'\n')
        valid_loss_min = np.mean(val_losses)
            
    return val_losses, val_acc, val_f1, vo, vp, valid_loss_min 