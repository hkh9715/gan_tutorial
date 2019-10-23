import argparse
import time
import torch
import torch.nn as nn
from loader import data_loader
from model import Generator,Discriminator
from losses import gd_loss
import torch.optim as optim
import torchvision.utils as vutils
from collections import defaultdict


parser = argparse.ArgumentParser()

parser.add_argument('--img_size', default=64, type=int)
parser.add_argument('--batchsize', default=2, type=int)
parser.add_argument('--dataroot', default=r'C:\\tensor_code\\GAN_code\\dataset\\celeba-dataset\\img_align_celeba', type=str)
parser.add_argument('--checkpoint_dir',default=r'C:\\tensor_code\\GAN_code\\DCGAN', type=str)
parser.add_argument('--val_dir',default=r'/content/drive/My Drive/LetsGAN/DCGAN/val_out', type=str)

parser.add_argument('--num_work', default=2, type=int)
parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--gpu_num', default="0", type=int)

parser.add_argument('--nz', default=100, type=int)
parser.add_argument('--ngf', default=64, type=int)
parser.add_argument('--ndf', default=64, type=int)
parser.add_argument('--nc', default=3, type=int)

parser.add_argument('--lr', default=0.0002, type=float)
parser.add_argument('--beta1', default=0.5, type=float)
parser.add_argument('--num_epochs', default=100, type=int)

parser.add_argument('--d_steps', default=2, type=int)
parser.add_argument('--g_steps', default=2, type=int)

parser.add_argument('--print_every', default=1, type=int)
parser.add_argument('--val_epochs',default=1,type=int)
parser.add_argument('--checkpoint_every', default=100, type=int)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def main(args):
    dset,dataloader=data_loader(args)
    ngpu=args.gpu_num
    checkpoint_path=args.checkpoint_dir

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    netG=Generator(args).to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
    netG.apply(weights_init)
    #print(netG)

    netD = Discriminator(args).to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))
    netD.apply(weights_init)
    #print(netD)

    loss=gd_loss()
    gloss=loss.g_loss
    dloss=loss.d_loss

    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    num_epochs=args.num_epochs
    val_epochs=args.val_epochs
    dsteps=args.d_steps
    gsteps=args.g_steps

    val_dir=args.val_dir

    checkpoint={
        'G_losses':defaultdict(list),
        'D_losses':defaultdict(list),
        'g_state':None,
        'd_state':None,
        'epoch':defaultdict(list),
        'g_optim_state':None,
        'd_optim_state':None
    }


    for epoch in range(num_epochs):
        start=time.time()
        for i,batch in enumerate(dataloader):
            
            losses_d=discriminator_step(args,batch,netG,netD,dloss,optimizerD,device)
            losses_g=generator_step(args,batch,netG,netD,gloss,optimizerG,device)
            

            if i//args.checkpoint_every==0:
                checkpoint['D_losses'][i].append(losses_d['D_total_loss'])
                checkpoint['G_losses'][i].append(losses_g['G_loss'])
                checkpoint['epoch'][i].append(epoch)
                checkpoint['d_state']=netD.state_dict()
                checkpoint['d_optim_state'] = optimizerD.state_dict()
                checkpoint['g_state']=netG.state_dict()
                checkpoint['g_optim_state'] = optimizerG.state_dict()

            if i % args.print_every == 0:
                print(epoch, num_epochs, i, len(dataloader),
                    losses_d['D_total_loss'], losses_g['G_loss'], losses_d['D_real_loss'],time.time()-start)
                print('[%d/%d][%d/%d]\tLoss_D_total: %.4f\tLoss_G: %.4f\tTime: %.4f'
                % (epoch, num_epochs, i, len(dataloader),
                    losses_d['D_total_loss'], losses_g['G_loss'], losses_d['D_real_loss'],time.time()-start))
                start=time.time()
                

        if epoch//val_epochs==0:
            noise=z=torch.randn(1,args.nz,1,1,device=device)
            with torch.no_grad():
                fake_img=netG(z)
                vutils.save_image(fake_img.data,
                '%s/fake_samples_epoch_%s.png' % (val_dir, str(epoch)),
                normalize=True)
            torch.save(checkpoint, checkpoint_path)




def discriminator_step(args,batch,generator_,discriminator_,d_loss,opt_dis,device):

  
    losses={}
    label=torch.full((args.batchsize,),1,device=device)
    real_cpu=batch[0].to(device)
    b_size=real_cpu.size(0)

    discriminator_.zero_grad()
    output = discriminator_(real_cpu).view(-1)
    # Calculate loss on all-real batch
    errD_real = d_loss(output, label)
    # Calculate gradients for D in backward pass
    errD_real.backward()
    D_x = output.mean().item()

    label.fill_(0)
    z=torch.randn(args.batchsize,args.nz,1,1,device=device)
    fake=generator_(z)
    output=discriminator_(fake.detach()).view(-1)
    errD_fake=d_loss(output,label)  
    errD_fake.backward()
    D_G_z1 = output.mean().item()
    errD=errD_real+errD_fake

    opt_dis.step()

    losses['D_real_loss']=errD_real.item()
    losses['D_fake_loss']=errD_fake.item()
    losses['D_total_loss']=errD.item()
    
    return losses


def generator_step(args,batch,generator_,discriminator_,g_loss,opt_gen,device):

    losses={}
    label=torch.full((args.batchsize,),1,device=device)

    generator_.zero_grad()

    z=torch.randn(args.batchsize,args.nz,1,1,device=device)
    fake=generator_(z)
    output=discriminator_(fake).view(-1)
    errG=g_loss(output,label)
    errG.backward()
    D_G_z2 = output.mean().item()
    opt_gen.step()

    losses['G_loss']=errG.item()

    return losses

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
