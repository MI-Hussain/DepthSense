from datetime import datetime
import os
import math
import time
import torch
import torch.nn as nn
import numpy as np
import utils
import random
from tqdm import tqdm

from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from utils import PolynomialLRDecay
from dataloader.nusc_loader1 import NuScenesLoader, NuScenesLoader1
from loss import OrdinalRegressionLoss
from valid_loader import train_one_epoch, validation
    

# set arguments
BATCH_SIZE = 2
EPOCHS = 50
LR = 0.005
END_LR = 0.0001
POLY_POWER = 0.80
LR_PATIENCE = 5
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
MAX_ITER = 300000
WORKERS = 10
SEED = 1984
PRINT_FREQ = 100

SIZE = (350,800)
NSWEEPS = 5

RGB_ONLY = False
# min value (meter) for benchmark training set: 1.9766
# max value (meter) for benchmark training set: 90.4414
# min value (meter) for eigen training set: 0.704
# max value (meter) for eigen training set: 79.729


ORD_NUM = 80
GAMMA = 0.3
ALPHA = 1
BETA = 80

# set random seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# create output dir,
# _now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')n
# DIR_NAME = '{}_{}'.format('radar0_450_800',_now)
output_dir = os.path.join('./result','radar'.format(NSWEEPS, SIZE[0], SIZE[1]))
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'valid')
logdir = os.path.join(output_dir, 'log')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(val_dir):
    os.makedirs(val_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)
    
print('OUTPUT_DIR = {}'.format(output_dir))

# set dataloader
DATA_ROOT = '/home/mlv/work/data/sets/nuscenes'

train_set = NuScenesLoader(data_root=DATA_ROOT, mode='train', nsweeps=NSWEEPS)
val_set = NuScenesLoader1(data_root=DATA_ROOT)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)

# create model
from model import rvmde
model = rvmde.RVMDE(input_size=SIZE, pretrained=True) 

data=torch.load('./paper_weights_rvmde.pth')
model.load_state_dict(data)
model=model.cuda()

print('GPU number: {}'.format(torch.cuda.device_count()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# if GPU number > 1, then use multiple GPUs
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.cuda()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = PolynomialLRDecay(optimizer, max_decay_steps=MAX_ITER, end_learning_rate=END_LR, power=POLY_POWER)

# loss function
ord_loss = OrdinalRegressionLoss(ord_num=ORD_NUM, beta=BETA)
         

logger = SummaryWriter(logdir)
epochbar = tqdm(total=EPOCHS)

for epoch in range(EPOCHS):
    
    train_one_epoch(device, train_loader, model, train_dir, ord_loss, optimizer, epoch, logger, PRINT_FREQ, BETA=BETA, GAMMA=GAMMA, ORD_NUM=80.0, RGB_ONLY=RGB_ONLY)  
    
    validation(val_loader, model)
    
    # save model and checkpoint per epoch
    checkpoint_filename = os.path.join(output_dir, 'checkpoint-{}.pth.tar'.format(str(epoch)))
    torch.save(model, checkpoint_filename)
       
    epochbar.update(1)




