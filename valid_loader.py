"""
Parts of the code is borrowed from https://github.com/lochenchou/DORN_radar

For further details please visit https://github.com/lochenchou/DORN_radar
"""


import os
import time
import torch
import numpy as np
import utils
from tqdm import tqdm
from metrics import AverageMeter, Result, compute_errors
import torch.nn.functional as F
from torchvision import transforms
from loss import OrdinalRegressionLoss
from tensorboardX import SummaryWriter
import matplotlib.pylab as plt 



BATCH_SIZE = 1
WORKERS = 9
PRINT_FREQ = 250
in_size = (350,800)
n_sweeps = 5

ORD_NUM = 80
GAMMA = 0.3
ALPHA = 1
BETA = 80
epoch = 1

RGB_ONLY = False

inv_normalize = transforms.Normalize(
   mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
   std=[1/0.229, 1/0.224, 1/0.225]
)

# validation    

# set dataloader
DATA_ROOT = '/media/auto/9b845cf4-e3a3-4062-9101-8e58973d8f26/ISHFAQ/CARLA_SERVER_WORK_FOLDER_CAN_BUS_OBJECT_DETECTION_WORK/Ishfaq/data/sets/nuscenes'


output_dir = os.path.join('./result','radar'.format(n_sweeps, in_size[0], in_size[1]))
#train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'valid')
logdir = os.path.join(output_dir, 'log')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(val_dir):
    os.makedirs(val_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


logger = SummaryWriter(logdir)

# train
def train_one_epoch(device, train_loader, model, output_dir, ord_loss, optimizer, epoch, logger, PRINT_FREQ, BETA, GAMMA, ORD_NUM=80.0, RGB_ONLY=True):
    
    avg80_sparse = AverageMeter()
    avg80_dense = AverageMeter()
    
    model.train()  # switch to train mode

    iter_per_epoch = len(train_loader)
    trainbar = tqdm(total=iter_per_epoch)
    end = time.time()
    
    for i, data in enumerate(train_loader):
        _rgb, _sparse_depth, _dense_depth = data['RGB'].to(device), data['SPARSE'].to(device), data['DENSE'].to(device)
        _radar_depth = data['RADAR'].to(device)
        torch.cuda.synchronize()
        data_time = time.time() - end

        
        # compute output
        end = time.time()

        with torch.autograd.detect_anomaly():
            if RGB_ONLY:
                _pred_prob, _pred_label = model(_rgb) 
            else:
                _pred_prob, _pred_label = model(_rgb, _radar_depth)
                
            loss = ord_loss(_pred_prob, _dense_depth) # calculate ord loss with dense_depth
            optimizer.zero_grad()
            loss.backward()  
            optimizer.step()
            
        torch.cuda.synchronize()
        gpu_time = time.time() - end

        pred_depth = utils.label2depth_sid(_pred_label, K=ORD_NUM, alpha=1.0, beta=BETA, gamma=GAMMA)
        # calculate metrices with ground truth sparse depth
        s_abs_rel, s_sq_rel, s_rmse, s_rmse_log, s_a1, s_a2, s_a3 = compute_errors(_sparse_depth, pred_depth.to(device))
        d_abs_rel, d_sq_rel, d_rmse, d_rmse_log, d_a1, d_a2, d_a3 = compute_errors(_dense_depth, pred_depth.to(device))
        
        result80_sparse = Result()
        result80_sparse.evaluate(pred_depth, _sparse_depth.data, cap=80)
        avg80_sparse.update(result80_sparse, gpu_time, data_time, _rgb.size(0))
        
        result80_dense = Result()
        result80_dense.evaluate(pred_depth, _dense_depth.data, cap=80)
        avg80_dense.update(result80_dense, gpu_time, data_time, _rgb.size(0))

        end = time.time()

        # update progress bar and show loss
        trainbar.set_postfix(ORD_LOSS='{:.2f}||DENSE||RMSE={:.2f},delta={:.2f}/{:.2f}|||SPARSE||RMSE={:.2f},delta={:.2f}/{:.2f}|'.format(loss,d_rmse,d_a1,d_a2,s_rmse,s_a1,s_a2))
        trainbar.update(1)
        
        
        if (i + 1) % PRINT_FREQ == 0:
            print('SPARSE: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'RMSE_log={result.rmse_log:.3f}({average.rmse_log:.3f}) '
                  'AbsRel={result.absrel:.2f}({average.absrel:.2f}) '
                  'SqRel={result.squared_rel:.2f}({average.squared_rel:.2f}) '
                  'SILog={result.silog:.2f}({average.silog:.2f}) '
                  'iRMSE={result.irmse:.2f}({average.irmse:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'Delta2={result.delta2:.3f}({average.delta2:.3f}) '
                  'Delta3={result.delta3:.3f}({average.delta3:.3f})'.format(
                i + 1, len(train_loader), gpu_time=gpu_time, result=result80_sparse, average=avg80_sparse.average()))

            current_step = int(epoch*iter_per_epoch+i+1)
        
            if RGB_ONLY:
                img_merge = utils.batch_merge_into_row(_rgb, _dense_depth.data, pred_depth)
                filename = os.path.join(output_dir,'step_{}.png'.format(current_step))
                utils.save_image(img_merge, filename)
            else:
#                img_merge = utils.batch_merge_into_row(_rgb, _dense_depth.data, pred_depth)
                img_merge = utils.batch_merge_into_row_radar(_rgb, _radar_depth.data, _dense_depth.data, pred_depth)
                filename = os.path.join(output_dir,'step_{}.png'.format(current_step))
                utils.save_image(img_merge, filename)
            
            logger.add_scalar('TRAIN/SPARSE_RMSE', avg80_sparse.average().rmse, current_step)
            logger.add_scalar('TRAIN/SPARSE_RMSE_log', avg80_sparse.average().rmse_log, current_step)
            logger.add_scalar('TRAIN/SPARSE_iRMSE', avg80_sparse.average().irmse, current_step)
            logger.add_scalar('TRAIN/SPARSE_SILog', avg80_sparse.average().silog, current_step)
            logger.add_scalar('TRAIN/SPARSE_AbsRel', avg80_sparse.average().absrel, current_step)
            logger.add_scalar('TRAIN/SPARSE_SqRel', avg80_sparse.average().squared_rel, current_step)
            logger.add_scalar('TRAIN/SPARSE_Delta1', avg80_sparse.average().delta1, current_step)
            logger.add_scalar('TRAIN/SPARSE_Delta2', avg80_sparse.average().delta2, current_step)
            logger.add_scalar('TRAIN/SPARSE_Delta3', avg80_sparse.average().delta3, current_step)
            
            logger.add_scalar('TRAIN/DENSE_RMSE', avg80_dense.average().rmse, current_step)
            logger.add_scalar('TRAIN/DENSE_RMSE_log', avg80_dense.average().rmse_log, current_step)
            logger.add_scalar('TRAIN/DENSE_iRMSE', avg80_dense.average().irmse, current_step)
            logger.add_scalar('TRAIN/DENSE_SILog', avg80_dense.average().silog, current_step)
            logger.add_scalar('TRAIN/DENSE_AbsRel', avg80_dense.average().absrel, current_step)
            logger.add_scalar('TRAIN/DENSE_SqRel', avg80_dense.average().squared_rel, current_step)
            logger.add_scalar('TRAIN/DENSE_Delta1', avg80_dense.average().delta1, current_step)
            logger.add_scalar('TRAIN/DENSE_Delta2', avg80_dense.average().delta2, current_step)
            logger.add_scalar('TRAIN/DENSE_Delta3', avg80_dense.average().delta3, current_step)
            
            # reset average meter
            result80_sparse = Result()
            avg80_sparse = AverageMeter()   
            result80_dense = Result()
            avg80_dense = AverageMeter()        



i = 0
def validation(data_loader, model):
    #model.to(device)
    ord_loss = OrdinalRegressionLoss(ord_num=ORD_NUM, beta=BETA)

    avg80_sparse = AverageMeter()
    avg80_dense = AverageMeter()
    #print(model)
    
    model.eval()
    
    end = time.time()
    skip =int(len(data_loader)/10)
    img_list = []
    
    evalbar = tqdm(total=len(data_loader))
    
    for i, data in enumerate(data_loader):
        _rgb, _sparse_depth, _dense_depth = data['RGB'].to(device), data['SPARSE'].to(device), data['DENSE'].to(device)
        _radar_depth = data['RADAR'].to(device)


        torch.cuda.synchronize()
        data_time = time.time() - end
        #_rgb = inv_normalize(_rgb)
        
        # compute output
        end = time.time()
        with torch.no_grad():
            if RGB_ONLY:
                _pred_prob, _pred_label = model(_rgb) 
            else:
                _pred_prob, _pred_label = model(_rgb, _radar_depth)
                
            loss = ord_loss(_pred_prob, _dense_depth)
            
        torch.cuda.synchronize()
        gpu_time = time.time() - end
        
        pred_depth = utils.label2depth_sid(_pred_label, K=ORD_NUM, alpha=1.0, beta=BETA, gamma=GAMMA)
        s_abs_rel, s_sq_rel, s_rmse, s_rmse_log, s_a1, s_a2, s_a3 = compute_errors(_sparse_depth, pred_depth.to(device))
        d_abs_rel, d_sq_rel, d_rmse, d_rmse_log, d_a1, d_a2, d_a3 = compute_errors(_dense_depth, pred_depth.to(device))
        
        # measure accuracy and record loss
        result80_sparse = Result()
        result80_sparse.evaluate(pred_depth, _sparse_depth.data, cap=80)
        avg80_sparse.update(result80_sparse, gpu_time, data_time, _rgb.size(0))
        
        result80_dense = Result()
        result80_dense.evaluate(pred_depth, _dense_depth.data, cap=80)
        avg80_dense.update(result80_dense, gpu_time, data_time, _rgb.size(0))

        end = time.time()
        
        # save images for visualization 
        #if i == 0:
        i = i+1

        filename = os.path.join(output_dir,'eval_{}.png'.format(int(epoch+i)))
        if i == 600:
        	_rgb = inv_normalize(_rgb)
        	img_merge = utils.merge_into_row_with_radar(_rgb, _radar_depth, _dense_depth, pred_depth)
        	utils.save_image(img_merge, filename)


        # if (i + 1) % PRINT_FREQ == 0:
        #     print('Test: [{0}/{1}]\t'
        #           't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
        #           'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
        #           'RMSE_log={result.rmse_log:.3f}({average.rmse_log:.3f}) '
        #           'AbsRel={result.absrel:.2f}({average.absrel:.2f}) '
        #           'SqRel={result.squared_rel:.2f}({average.squared_rel:.2f}) '
        #           'SILog={result.silog:.2f}({average.silog:.2f}) '
        #           'iRMSE={result.irmse:.2f}({average.irmse:.2f}) '
        #           'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
        #           'Delta2={result.delta2:.3f}({average.delta2:.3f}) '
        #           'Delta3={result.delta3:.3f}({average.delta3:.3f})'.format(
        #         i + 1, len(data_loader), gpu_time=gpu_time, result=result80_sparse, average=avg80_sparse.average()))
            
        # update progress bar and show loss
        evalbar.set_postfix(ORD_LOSS='{:.2f}||DENSE||RMSE={:.2f},delta={:.2f}/{:.2f}|||SPARSE||RMSE={:.2f},delta={:.2f}/{:.2f}|'.format(loss,d_rmse,d_a1,d_a2,s_rmse,s_a1,s_a2))
        evalbar.update(1)

        i = i+1

    print('\n**** EVALUATE WITH SPARSE DEPTH ****\n'
          '\n**** CAP=80 ****\n'
          'RMSE={average.rmse:.3f}\n'
          'RMSE_log={average.rmse_log:.3f}\n'
          'AbsRel={average.absrel:.3f}\n'
          'SqRel={average.squared_rel:.3f}\n'
          'SILog={average.silog:.3f}\n'
          'Delta1={average.delta1:.3f}\n'
          'Delta2={average.delta2:.3f}\n'
          'Delta3={average.delta3:.3f}\n'
          'iRMSE={average.irmse:.3f}\n'
          'iMAE={average.imae:.3f}\n'
          't_GPU={average.gpu_time:.3f}\n'.format(
        average=avg80_sparse.average()))
    
    logger.add_scalar('VAL_CAP80/SPARSE_RMSE', avg80_sparse.average().rmse, epoch)
    logger.add_scalar('VAL_CAP80/SPARSE_RMSE_log', avg80_sparse.average().rmse_log, epoch)
    logger.add_scalar('VAL_CAP80/SPARSE_iRMSE', avg80_sparse.average().irmse, epoch)
    logger.add_scalar('VAL_CAP80/SPARSE_SILog', avg80_sparse.average().silog, epoch)
    logger.add_scalar('VAL_CAP80/SPARSE_AbsRel', avg80_sparse.average().absrel, epoch)
    logger.add_scalar('VAL_CAP80/SPARSE_SqRel', avg80_sparse.average().squared_rel, epoch)
    logger.add_scalar('VAL_CAP80/SPARSE_Delta1', avg80_sparse.average().delta1, epoch)
    logger.add_scalar('VAL_CAP80/SPARSE_Delta2', avg80_sparse.average().delta2, epoch)
    logger.add_scalar('VAL_CAP80/SPARSE_Delta3', avg80_sparse.average().delta3, epoch)
    
    logger.add_scalar('VAL_CAP80/DENSE_RMSE', avg80_dense.average().rmse, epoch)
    logger.add_scalar('VAL_CAP80/DENSE_RMSE_log', avg80_dense.average().rmse_log, epoch)
    logger.add_scalar('VAL_CAP80/DENSE_iRMSE', avg80_dense.average().irmse, epoch)
    logger.add_scalar('VAL_CAP80/DENSE_SILog', avg80_dense.average().silog, epoch)
    logger.add_scalar('VAL_CAP80/DENSE_AbsRel', avg80_dense.average().absrel, epoch)
    logger.add_scalar('VAL_CAP80/DENSE_SqRel', avg80_dense.average().squared_rel, epoch)
    logger.add_scalar('VAL_CAP80/DENSE_Delta1', avg80_dense.average().delta1, epoch)
    logger.add_scalar('VAL_CAP80/DENSE_Delta2', avg80_dense.average().delta2, epoch)
    logger.add_scalar('VAL_CAP80/DENSE_Delta3', avg80_dense.average().delta3, epoch)
    
 
