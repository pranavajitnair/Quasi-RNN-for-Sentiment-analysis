import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model import Classifier
from data_loader import read_file_names,get_raw_data,preprocess,get_batches,DataLoader

import gensim.models as gs

def train(model,dataloader,epochs):
        optimizer=optim.Adam(model.parameters(),lr=1)
        lossFunction=nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
                model.train()
                print(model.qrnn1.convolution.weight.grad)
                optimizer.zero_grad()
                
                x,y,mask=dataloader.load_next_batch(True)
                
                logits=model(x,mask)
                loss=lossFunction(logits,y)
                training_loss=loss.item()
                print(loss)
                loss.backward()
                optimizer.step()
                print(model.padding.grad)
                
                model.eval()
                x,y,mask=dataloader.load_next_batch(False)
                
                logits=model(x,mask)
                lossv=lossFunction(logits,y)
                validation_loss=lossv.item()
                
                _,indices=torch.max(F.softmax(logits,dim=1),dim=1)
                indices+=1
                correct=torch.sum(indices==y)
                
                print(y,indices)
                print('epoch=',epoch+1,'training loss=',training_loss,'validation loss=',validation_loss,'validation accuracy=',int(correct*5))
                

batch_size=20
kernel_size=2

embedding_size=50
hidden_size=100
epochs=100

filenames=read_file_names()
x_train,y_train=get_raw_data(filenames)

x_train,_=preprocess(x_train)
word_dict=gs.Word2Vec(x_train,min_count=1,size=embedding_size)

x_train,y_train,masks=get_batches(x_train,y_train,batch_size)

Model=Classifier(embedding_size,hidden_size,kernel_size)
for name,_ in Model.named_parameters():
        print(name)

dataLoader=DataLoader(x_train,y_train,masks,word_dict,embedding_size,Model.padding)

train(Model,dataLoader,epochs)