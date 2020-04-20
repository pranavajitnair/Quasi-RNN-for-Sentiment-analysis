import torch
import torch.nn as nn


class QRNN(nn.Module):
        def __init__(self,embedding_size,hidden_size,kernel_size=2):
                super(QRNN,self).__init__()
                
                self.embedding_size=embedding_size
                self.hidden_size=hidden_size
                self.kernel_size=kernel_size
                
                self.convolution=nn.Linear(kernel_size*embedding_size,3*hidden_size)
                self.broadcast=nn.Linear(hidden_size,3*hidden_size)
                
        def connvolution_step(self,input,hidden=None):
                x_temp=input[:,:input.shape[1]-1,:]
                x_temp=torch.cat((torch.zeros(input.shape[0],1,input.shape[2]),x_temp),dim=1)
                
                input=torch.cat((input,x_temp),dim=2)
                output=self.convolution(input)
                
                if hidden is not None:
                        output=output+self.broadcast(hidden)
                        
                cell,f,o=output[:,:,:self.hidden_size],output[:,:,self.hidden_size:2*self.hidden_size],output[:,:,2*self.hidden_size:]
                
                return torch.tanh(cell),torch.sigmoid(f),torch.sigmoid(o)
            
        def recurrent_step(self,o,f,z,c):
            
                if c is None:
                        c_prime=(1-f)*z
                        h=o*c_prime
                else:
                        c_prime=f*c+(1-f)*z
                        h=o*c_prime
                        
                return c_prime,h
        
        def forward(self,input,mask,c=None,h=None):
                cell,f,o=self.connvolution_step(input,h)
                
                batch_size=input.shape[0]
                
#                output=torch.zeros(batch_size,input.shape[1],self.hidden_size)
                cell_state=torch.zeros(batch_size,1,self.hidden_size)
                hidden=torch.zeros(batch_size,1,self.hidden_size)
                
                hidden_states=[]
                cell_states=[]
                
                for i in range(input.shape[1]):
                        O=o[:,i,:].view(batch_size,1,-1)
                        F=f[:,i,:].view(batch_size,1,-1)
                        Cell=cell[:,i,:].view(batch_size,1,-1)
                        
                        c,h=self.recurrent_step(O,F,Cell,c)
                        
                        cell_states.append(c)
                        hidden_states.append(h)
                
                hidden_temp=torch.cat(hidden_states,dim=1)
                cell_temp=torch.cat(cell_states,dim=1)
                
                for i in range(batch_size):
#                        output[i,:mask[i],:]=hidden_temp[i,:mask[i],:]
                        cell_state[i,0,:]=cell_temp[i,mask[i]-1,:]
                        hidden[i,0,:]=hidden_temp[i,mask[i]-1,:]
                        
                return hidden_temp,hidden,cell_state