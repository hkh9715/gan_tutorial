import torch
import torch.nn as nn
import random

class gd_loss:
    def __init__(self):
        self.criterion = nn.BCELoss()
    def g_loss(self,output,label):
        return self.criterion(output,label)
    def d_loss(self,output,label):
        return self.criterion(output,label)