# shared: train loop, metrics, checkpointing
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def train_epoch(model,optimizer,criterion,loader,device):
    model.train()
    correct = 0
    total = 0
    total_loss = 0

    for x,y in loader:
        x,y = x.to(device),y.to(device)
        optimizer.zero_grad() # zero out the gradients from the previous step, otherwise they will accumulate and affect the current step
        out = model(x) # save the output 
        loss = criterion(out,y) # calculate the loss 
        loss.backward() 
        optimizer.step() # update the model parameters based on the calculated gradients

        total_loss += loss.item()
        preds = torch.argmax(out,dim = 1)
        correct += (preds==y).sum().item()
        total += y.size(0)
        
    accuracy = correct/total
    average_loss = total_loss/len(loader)
    return average_loss , accuracy

def eval_epoch(model, loader, criterion, device): 
    model.eval()
    total = 0
    correct = 0
    total_loss = 0
    all_preds = []
    all_labels = []
    for x,y in loader:
        x,y = x.to_device(),y.to_device()
        out = model(x)
        loss = criterion(out,y)
        total_loss += loss.item()
        preds = torch.argmax(out,dim=1)
        correct += (preds==y).sum().item()
        total += y.size(0)
        all_preds.extend(preds.cpu().numpy()) 
        all_labels.extend(y.cpu().numpy())  
    acc = correct/total
    
    return total_loss/len(loader), acc, all_preds, all_labels

def save_checkpoint(model, optimizer, epoch, val_acc, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'val_acc': val_acc
    }, path)
    
def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_acc = checkpoint['val_acc']
    return model, epoch, val_acc

def plot_confusion_matrix(preds, labels, class_names):
    cm = confusion_matrix(preds,labels)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("Predicted")
    plt.xlabel("True")
    plt.title("Confusion Matrix")
    plt.show()

def compute_metrics(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)

    acc = (preds == labels).mean()

    f1_macro = f1_score(labels, preds, average="macro")
    f1_per_class = f1_score(labels, preds, average=None)

    return {
        "accuracy": acc,
        "macro_f1": f1_macro,
        "per_class_f1": f1_per_class
    }
    