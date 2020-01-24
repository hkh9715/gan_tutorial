import torch
import torch.nn as nn
from torchvision import datasets, transforms
import argparse
from model import generator,discriminator
import os
from utils import *


if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--img_size', default=64, type=int)
    parser.add_argument('--batchsize', default=64, type=int)
    parser.add_argument('--dataroot', default=r'C:\\tensor_code\\GAN_code\\dataset\\fmnist', type=str)
    parser.add_argument('--checkpoint_dir',default=r'C:\\tensor_code\\GAN_code\\DCGAN', type=str)
    parser.add_argument('--val_dir',default=r'/content/drive/My Drive/LetsGAN/DCGAN/val_out', type=str)

    parser.add_argument('--num_work', default=2, type=int)
    parser.add_argument('--use_gpu', default=1, type=int)
    parser.add_argument('--gpu_num', default="0", type=int)

    parser.add_argument('--nz', default=100, type=int) #z 벡터 차원
    parser.add_argument('--nc', default=10, type=int) #원핫인코딩수

    parser.add_argument('--d_lr', default=0.0001, type=float)
    parser.add_argument('--g_lr', default=0.0002, type=float)
    parser.add_argument('--beta', type=float, default=0.5, help='beta for adam optimizer')
    parser.add_argument('--beta1', type=float, default=0.999, help='beta1 for adam optimizer')
    parser.add_argument('--num_epochs', default=100, type=int)

    parser.add_argument('--d_steps', default=2, type=int)
    parser.add_argument('--g_steps', default=2, type=int)

    parser.add_argument('--print_every', default=100, type=int)
    parser.add_argument('--val_epochs',default=1,type=int)
    parser.add_argument('--checkpoint_every', default=100, type=int)
    

    args = parser.parse_args()

    batch_size=args.batchsize
    epochs=args.num_epochs

    data_dir=args.dataroot
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (1.0,))])
    
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
        trainset = datasets.FashionMNIST(data_dir, download=True, train=True,transform=transform)

    trainset = datasets.FashionMNIST(data_dir, download=False, train=True,transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)    

    G=generator(args)
    D=discriminator(args)

    criterion = torch.nn.BCELoss()

    d_optimizer = torch.optim.Adam(D.parameters(), lr=args.d_lr, betas=(args.beta, args.beta1))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=args.d_lr, betas=(args.beta, args.beta1))


    for i in range(epochs): 
        for idx,data in enumerate(trainloader):
            

            x=data[0].to(device)
            y=data[1].to(device)

            g_optimizer.zero_grad()
            fake=G(y)

            real_output=D(x,y)
            fake_output=D(fake,y)

            real_label=torch.ones(batch_size).float()
            fake_label=torch.zeros(batch_size).float()

            g_loss = criterion(fake_output, real_label)


            g_loss.backward() 
            g_optimizer.step()

            d_optimizer.zero_grad()

            real_loss=criterion(real_output,real_label)
            fake_loss=criterion(fake_output.detach(),fake_label)

            d_loss=real_loss+fake_loss

            d_loss.backward()
            d_optimizer.step()

            if (idx+1)%10==0:
                print("Epoch: %d step %d G_Loss: %f D_Loss: %f" %(i,idx,g_loss.item(),d_loss.item()))




        



