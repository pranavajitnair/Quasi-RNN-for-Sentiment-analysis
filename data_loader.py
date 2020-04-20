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

class DataLoader(object):
        def __init__(self,datax,datay,word_dict):
            
                self.datax=datax
                self.datay=datay
                
                self.word_dict=word_dict
                
                self.counter=0
                self.len=len(datax)//2
                
                self.counter2=0
                
        def load_next_batch(self,train):
                if train:
                        x=self.datax[self.counter]
                        y=self.datay[self.counter]
                        self.counter=(self.counter+1)%self.len
                        
                else:
                        x=self.datax[self.len+self.counter2]
                        y=self.datay[self.len+self.counter2]
                        self.counter2=(1+self.counter2)%20
                        
                l=[]
                for word in x:
                        l.append(self.word_dict[word])
                
                l=torch.tensor(l)
                y=torch.tensor(y).view(1)
                
                return l,y
