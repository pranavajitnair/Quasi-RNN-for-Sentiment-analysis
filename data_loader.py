import os
import re
import torch

def read_file_names():       
        filenames=[]
        
        for _,_,filename in os.walk('/home/pranav/ml/data/aclImdb/train/pos'):
                for names in filename:
                        filenames.append(names)
                        
        return filenames

def get_raw_data(filenames):        
        x_train=[]
        y_train=[]
        
        for filename in filenames:
                with open('/home/pranav/ml/data/aclImdb/train/pos/'+filename, 'r') as f:
                        corpus = f.readlines()
                        x_train.append(corpus)
                        y_train.append(int(filename[-5]))
                        
        return x_train,y_train
    
def preprocess(data):
        x1=[]
        ma=0
        
        for x_t in data:
            
                x=x_t[0]
                x=re.sub(r'(\w)\..',r"\1",x)
                x=re.sub(r'(\w)\...',r"\1",x)
                x=re.sub(r'(\w)\..',r"\1",x)
                x=re.sub("(\w)\)",r"\1 )",x)
                x=re.sub("(\))(\W)",r") \2",x)
                x=re.sub("(\))(\w)",r") \2",x)
                x=re.sub("(\W)\(",r"\1 (",x)
                x=re.sub("(\w)\(",r"\1 (",x)
                x=re.sub("(\()(\W)",r"( \2",x)
                x=re.sub("(\()(\w)",r"( \2",x)
                x=re.sub(r'(\w)\.',r"\1 .",x)
                x=re.sub(r'(\W)\.',r"\1 .",x)
                x=re.sub(r'\.(\w)',r". \1",x)
                x=re.sub(r'\.(\W)',r". \1",x)
                x=re.sub(r'!{2,}',r'!',x)
                x=re.sub(r'\?{2,}',r'?',x)
                x=re.sub(r'(\w)\?',r"\1 ?",x)
                x=re.sub(r'(\w)\!',r"\1 !",x)
                x=re.sub(r'(\w)\..',r"\1",x)
                x=re.sub(r'(\w)\...',r"\1",x)
                x=re.sub(r'(\?!){2,}',r'?!',x)
                x=re.sub(r'(!\?){2,}',r'!?',x)
                x=re.split(',| |<br|>|/|####|-|\*|:|;',x)
                x_prime=[]
                
                for word in x:
                        if word!='':
                                x_prime.append(word)
                                
                x1.append(x_prime)
                ma=max(ma,len(x_prime))
                
        return x1,ma
    
def get_batches(datax,datay,batch_size):
        batchesx=[]
        batchesy=[]
        masks=[]
        
        for i in range(len(datax)//batch_size):
                batchx=[]
                batchy=[]
                batchfinalx=[]
                ma=0
                mask=[]
                
                for j in range(i*batch_size,i*batch_size+batch_size):
                        batchx.append(datax[j])
                        batchy.append(datay[j])
                        ma=max(ma,len(datax[j]))
                        
                for sentence in batchx:
                        mask.append(len(sentence))
                        for _ in range(ma-len(sentence)):
                                sentence.append(0)
                        batchfinalx.append(sentence)                 
                        
                batchesx.append(batchfinalx)
                batchesy.append(batchy)
                masks.append(mask)
                
        return batchesx,batchesy,masks        


class DataLoader(object):
        def __init__(self,datax,datay,masks,word_dict,embed_dim,padding):
            
                self.datax=datax
                self.datay=datay
                self.masks=masks
                
                self.embed_dim=embed_dim
                self.word_dict=word_dict
                
                self.counter=0
                self.len=len(datax)//2
                
                self.padding=padding
                
        def load_next_batch(self,train):
                if train:
                        x=self.datax[self.counter]
                        y=self.datay[self.counter]
                        mask=self.masks[self.counter]
                        self.counter=(self.counter+1)%self.len
                        
                else:
                        x=self.datax[self.len]
                        y=self.datay[self.len]
                        mask=self.masks[self.len]
                
                l=[]
                for sentence in x:
                        sent=[]
                        
                        for word in sentence:
                            
                                if word!=0:
                                        sent.append(torch.tensor(self.word_dict[word]).view(1,-1))
                                else:
                                        sent.append(self.padding.view(1,-1))
                                        
                        sent=torch.cat(sent,dim=0)                
                        l.append(sent.view(1,-1,self.embed_dim))
                
                l=torch.cat(l,dim=0)
                y=torch.tensor(y)
                
                return l,y,mask