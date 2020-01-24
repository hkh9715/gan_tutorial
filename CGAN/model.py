import torch
import torch.nn as nn
import torch.nn.functional as F

class generator(nn.Module):

    def __init__(self,args):

        super(generator, self).__init__()
        
        self.nz=args.nz
        self.nc=args.nc  #원핫인코딩수
        in_d=self.nz+self.nc
        self.args=args

        self.label_emb=nn.Embedding(self.nc,self.nc)
        self.main = nn.Sequential(
                                  nn.ConvTranspose2d(in_d,in_d*8, kernel_size=14,padding=0), #4*4
                                  nn.BatchNorm2d(in_d*8),
                                  nn.ReLU(True),
                                  nn.ConvTranspose2d(in_d*8,1,kernel_size=15,padding=0), #4*4
                                  nn.BatchNorm2d(1),
                                  nn.ReLU(True)
                                    )
        
    def forward(self,c):

        a=self.label_emb(c)
        latent_z=torch.randn((self.args.batchsize,self.nz))


        in_v=torch.cat([latent_z,a],dim=1)
        out=self.main(in_v.unsqueeze(2).unsqueeze(3))

        return out

class discriminator(nn.Module):

    def __init__(self,args):

        super(discriminator, self).__init__()

        self.args=args
        self.nc=self.args.nc

        self.label_emb=nn.Embedding(self.nc,self.nc)
        self.l1=nn.Linear(28*28+10,1)

        
        #self.sig=nn.Sigmoid()

        # self.main=nn.Sequential(
        #                         nn.Conv2d(1, 16,kernel_size=3,stride=1,padding=1, bias=False), #16*28*28
        #                         nn.LeakyReLU(0.2, inplace=True),            
        #                         nn.Conv2d(16, 8,kernel_size=3,stride=1, padding=1, bias=False), #8*28*28
        #                         nn.LeakyReLU(0.2, inplace=True),
        #                         # nn.Linear(8*28*28,16)                       
        #                         )


    def forward(self,x,c):
        
        x=x.view(self.args.batchsize,-1)
        a=self.label_emb(c)

        x=torch.cat([x,a],dim=1)
        
        output=F.sigmoid(self.l1(x.unsqueeze(1)))
        
        return output


        





