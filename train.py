import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model import Classifier
from data_loader import read_file_names,get_raw_data,preprocess,DataLoader

import gensim.models as gs

def train(model,dataloader,epochs):
        optimizer=optim.Adam(model.parameters(),lr=0.01)
        lossFunction=nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                
                loss=0
                training_loss=0
                
                for _ in range(500):
                        x,y=dataloader.load_next_batch(True)
                        logits=model(x)
                        loss+=lossFunction(logits,y)
                        training_loss+=loss.item()

                loss.backward()
                optimizer.step()
                
                model.eval()
                vl=0
                acc=0
                
                for _ in range(20):
                        x,y=dataloader.load_next_batch(False)
                        
                        logits=model(x)
                        lossv=lossFunction(logits,y)
                        vl+=lossv.item()
                        
                        _,indices=torch.max(F.softmax(logits,dim=1),dim=1)
                        indices+=1
                        acc+=int(torch.sum(indices==y))
                
                print('epoch=',epoch+1,'training loss=',training_loss/500,'validation loss=',vl/20,'validation accuracy=',acc*5)
 
               
kernel_size=2
embedding_size=50
hidden_size=100
epochs=200

filenames=read_file_names()
x_train,y_train=get_raw_data(filenames)

x_train,_=preprocess(x_train)
word_dict=gs.Word2Vec(x_train,min_count=1,size=embedding_size)

Model=Classifier(embedding_size,hidden_size,kernel_size)

dataLoader=DataLoader(x_train,y_train,word_dict)

train(Model,dataLoader,epochs)
