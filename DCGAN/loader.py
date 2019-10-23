import torchvision.datasets as dset
import torch.utils.data
import torchvision.transforms as transforms

def data_loader(args,dataroot=r'C:\\tensor_code\\GAN_code\\dataset\\celeba-dataset\\img_align_celeba'):

    dataset = dset.ImageFolder(root=args.dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(args.img_size),
                               transforms.CenterCrop(args.img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batchsize,
                                         shuffle=True,
                                         num_workers=args.num_work,drop_last=True)
    return dataset,dataloader