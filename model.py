import torch.nn as nn
from qrnn import QRNN


class Classifier(nn.Module):
        def __init__(self,embedding_size,hidden_size,kernel_size=2):
            super(Classifier,self).__init__()
            
            self.qrnn1=QRNN(embedding_size,hidden_size,kernel_size)
            self.qrnn2=QRNN(hidden_size,hidden_size,kernel_size)
            self.qrnn3=QRNN(hidden_size,hidden_size,kernel_size)
            self.qrnn4=QRNN(hidden_size,hidden_size,kernel_size)
            
            self.classifier=nn.Linear(hidden_size,10)
            
        def forward(self,input,mask):
            
            output1,h,c=self.qrnn1(input,mask)
            output2,h,c=self.qrnn2(output1,mask,c,h)
            output3,h,c=self.qrnn3(output2,mask,c,h)
            output4,h,c=self.qrnn4(output3,mask,c,h)
            
            return self.classifier(h).squeeze(1)