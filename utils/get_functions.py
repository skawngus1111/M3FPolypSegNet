import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

def get_deivce() :
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.cuda.current_device()
    print("You are using \"{}\" device.".format(device))

    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_optimizer(optimizer, model, lr, momentum, weight_decay, **kwargs):
    params = list(filter(lambda p: p.requires_grad, model.parameters()))

    if 'FSDA' in kwargs.keys():
        params += list(filter(lambda p: p.requires_grad, kwargs['fsda'].parameters()))

    if optimizer == 'SGD' :
        optimizer = optim.SGD(params=params, lr=lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    elif optimizer == 'Adam' :
        optimizer = optim.Adam(params=params, lr=lr, weight_decay=weight_decay)
    elif optimizer == 'AdamW' :
        optimizer = optim.AdamW(params=params, lr=lr, weight_decay=weight_decay)
    else :
        print("Wrong optimizer")
        sys.exit()

    return optimizer

def get_lr(step, total_steps, lr_max, lr_min):
  """Compute learning rate according to cosine annealing schedule."""
  return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

def get_scheduler(LRS_name, optimizer, epochs, train_loader_len, learning_rate) :
    if LRS_name == 'SLRS' : # step learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
    elif LRS_name == 'MSLRS': # multi-step learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    elif LRS_name == 'CALRS': # cosine learning rate scheduler
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
                step,
                epochs * train_loader_len,
                1,  # lr_lambda computes multiplicative factor
                1e-6 / learning_rate))
    else :
        scheduler = None

    return scheduler

def get_criterion(criterion) :
    if criterion == 'CCE' :
        criterion = nn.CrossEntropyLoss()
    elif criterion == 'BCE' :
        criterion = nn.BCEWithLogitsLoss()
    elif criterion == 'MLSM' :
        criterion = nn.MultiLabelSoftMarginLoss()
    else :
        print("Wrong criterion")
        sys.exit()

    return criterion

def get_save_path(args):
    save_model_path = '{}_{}x{}_{}_{}_{}({}_{})_{}_{}'.format(args.data_type,
                                                              str(args.image_size), str(args.image_size),
                                                              str(args.batch_size),
                                                              args.model_name,
                                                              args.optimizer_name,
                                                              args.lr,
                                                              str(args.final_epoch).zfill(3),
                                                              str(args.LRS_name),
                                                              str(args.final_fsr))

    model_dirs = os.path.join(args.save_path, save_model_path)
    if not os.path.exists(os.path.join(model_dirs, 'model_weights')): os.makedirs(os.path.join(model_dirs, 'model_weights'))
    if not os.path.exists(os.path.join(model_dirs, 'test_reports')): os.makedirs(os.path.join(model_dirs, 'test_reports'))

    return model_dirs