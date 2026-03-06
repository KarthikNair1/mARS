import dataset
import utils
from utils import EarlyStopping, LRScheduler
from logging_utils_3D import calculate_metrics_from_probs, log_metrics, log_classification_plots, log_probability_histogram
import logging_utils_3D

import os
import pandas as pd
import argparse
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import time
import random
import wandb
import tqdm
import shutil
import yaml
import sys
from datetime import datetime 

def load_config():
    if len(sys.argv) < 2:
        print("Usage: python script.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def save_copy_of_config(filename):
    config_path = sys.argv[1]
    if not os.path.exists(args["output"]):
        os.makedirs(args["output"])

    dest_path = os.path.join(args["output"], filename)
    shutil.copyfile(config_path, dest_path)
    print(f"Configuration file copied to {dest_path}")

def set_defaults_config():
    default_dict = {
        'postfilter_fwhm_xy_mm': None,
        'normalize_mean': None,
        'normalize_std': None,
        'checkpoint': None,
        'resume': False,
        'lr_anneal': 15,
        'momentum': 0.9,
        'wd': 1e-4,
        'fast_dev_run': False,
        'WANDB_tags': [],
        'WANDB_run_name': None,
        'log_mipview_aggregation': True,
        'log_example_images': True,
        'BLUR_kernel_size': 5,
        'BLUR_sigma_low': 0.1,
        'BLUR_sigma_high': 2.0,
        'postfilter_fwhm_xy_mm_min': None,
        'postfilter_fwhm_xy_mm_max': None
    }

    for key, value in default_dict.items():
        if key not in args:
            args[key] = value

def init_wandb(config):
    wandb.login()
    run = wandb.init(project="MARS", name = args["WANDB_run_name"], 
                    tags = args["WANDB_tags"], config=config, save_code=True)

    # define metrics
    #wandb.define_metric("train/*", step_metric="train/step")
    #wandb.define_metric("val/*", step_metric="val/step")

    wandb.define_metric("epoch/train/*", step_metric="epoch")
    wandb.define_metric("step/train/*", step_metric="train_step")
    wandb.define_metric("epoch/val/*", step_metric="epoch")
    wandb.define_metric("mipview/*", step_metric="epoch")
    return run, wandb.config

def main():
    # Get user input
    global args
    args = load_config()
    set_defaults_config()

    # save config file
    t0 = time.time()
    datetimestr = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
    config_outname = "config_" + datetimestr + ".yaml"
    save_copy_of_config(config_outname)

    log_file = open(os.path.join(args["output"], f'log_{datetimestr}.log'),"w")
    sys.stdout = log_file
    sys.stderr = log_file

    global wandb_run
    wandb_run, args = init_wandb(args)

    print(args)
    best_auc = 0.

    # seed
    torch.manual_seed(2025)
    np.random.seed(2025)
    random.seed(2025)

    # Output directory and files
    if not os.path.isdir(args["output"]):
        try:
            os.mkdir(args["output"])
        except OSError:
            print ('Creation of the output directory "{}" failed.'.format(args["output"]))
        else:
            print ('Successfully created the output directory "{}".'.format(args["output"]))

    # Get model
    model = utils.get_model(args["arch"])

    if args["checkpoint"]:
        ch = torch.load(args["checkpoint"])
        # model.load_state_dict(ch['state_dict'])
        model = utils.model_prediction(args["checkpoint"])
        model.cuda()
    elif args["checkpoint_folder"]:
        bestrun = pd.read_csv(os.path.join(args["checkpoint_folder"],'best_run.csv')).run.values[0]
        ch = torch.load( os.path.join(args["checkpoint_folder"],'checkpoint_seed{}_run{}.pth'.format(args["seed_index"],bestrun)))
        # model.load_state_dict(ch['state_dict'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in ch['state_dict'].items() if k in model_dict}
        print('Loaded [{}/{}] keys from checkpoint'.format(len(pretrained_dict),len(model_dict)))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    if args["resume"]:
        ch = torch.load( os.path.join(args["output"],'checkpoint_seed'+str(args["seed_index"])+'_run'+str(args["run"])+'.pth') )
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in ch['state_dict'].items() if k in model_dict}
        print('Loaded [{}/{}] keys from checkpoint'.format(len(pretrained_dict),len(model_dict)))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    # Set optimizer
    optimizer = utils.create_optimizer(model, args["optimizer"], args["lr"], args["momentum"], args["wd"])
    if args["resume"] and 'optimizer' in ch:
        optimizer.load_state_dict(ch['optimizer'])
        print('Loaded optimizer state')
    cudnn.benchmark = True
    
    # Set datasets
    if args["augm"]==0:
        transform = None
    elif args["augm"]==1:
        transform = transforms.Compose([dataset.RandomFlip()])
    
    #train_dset,trainval_dset,val_dset,_,balance_weight_neg_pos = dataset.get_datasets_3d(transform,args.normalize,args.balance,args.seed_index)
    train_dset = dataset.get_datasets_3d(dataset_type='train', 
        norm=args["normalize"], transform=transform, 
        data_path_img = args["data_path_img"], data_path = args["train_datapath_csv"],
        postfilter_fwhm_xy_mm=args["postfilter_fwhm_xy_mm"],
        normalize_mean=args["normalize_mean"], normalize_std=args["normalize_std"],
        target_label_name=args["target_label_name"],
        postfilter_fwhm_xy_mm_min = args["postfilter_fwhm_xy_mm_min"],
        postfilter_fwhm_xy_mm_max = args["postfilter_fwhm_xy_mm_max"]
    )
    val_dset = dataset.get_datasets_3d(dataset_type='val', 
        norm=args["normalize"], transform=None, 
        data_path_img = args["data_path_img"], data_path = args["val_datapath_csv"],
        postfilter_fwhm_xy_mm=args["postfilter_fwhm_xy_mm"],
        normalize_mean=args["normalize_mean"], normalize_std=args["normalize_std"],
        target_label_name=args["target_label_name"],
        postfilter_fwhm_xy_mm_min=None, postfilter_fwhm_xy_mm_max=None
    )
    train_dset_for_evaluation = dataset.get_datasets_3d(dataset_type='val', 
        norm=args["normalize"], transform=None, 
        data_path_img = args["data_path_img"], data_path = args["train_datapath_csv"],
        postfilter_fwhm_xy_mm=args["postfilter_fwhm_xy_mm"],
        normalize_mean=args["normalize_mean"], normalize_std=args["normalize_std"],
        target_label_name=args["target_label_name"],
        postfilter_fwhm_xy_mm_min=None, postfilter_fwhm_xy_mm_max=None
    )

    print('Datasets train:{}, val:{}'.format(len(train_dset.df),len(val_dset.df))) 
    
    # Set loss criterion
    if args["balance"]:
        raise NotImplementedError("Balance option not implemented.")
        w = torch.Tensor(balance_weight_neg_pos)
        criterion = nn.BCEWithLogitsLoss(pos_weight=w).cuda()
    else:
        criterion = nn.BCEWithLogitsLoss().cuda()
    
    # Early stopping
    if args["lr_scheduler"]:
        print('INFO: Initializing learning rate scheduler')
        lr_scheduler = LRScheduler(optimizer)
        if args["resume"] and 'lr_scheduler' in ch:
            lr_scheduler.lr_scheduler.load_state_dict(ch['lr_scheduler'])
            print('Loaded lr_scheduler state')
    if args["early_stopping"]:
        print('INFO: Initializing early stopping')
        early_stopping = EarlyStopping()
        if args["resume"] and 'early_stopping' in ch:
            early_stopping.best_loss = ch['early_stopping']['best_loss']
            early_stopping.counter = ch['early_stopping']['counter']
            early_stopping.min_delta = ch['early_stopping']['min_delta']
            early_stopping.patience = ch['early_stopping']['patience']
            early_stopping.early_stop = ch['early_stopping']['early_stop']
            print('Loaded early_stopping state')
        
    # Set loaders
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=args["batch_size"], shuffle=True, num_workers=args["workers"])
    train_for_evaluation_loader = torch.utils.data.DataLoader(train_dset_for_evaluation, batch_size=args["batch_size"], shuffle=False, num_workers=args["workers"])
    val_loader = torch.utils.data.DataLoader(val_dset, batch_size=args["batch_size"], shuffle=False, num_workers=args["workers"])

    # Set output files
    convergence_name = 'convergence_seed'+str(args["seed_index"])+'_run'+str(args["run"])+'.csv'
    if not args["resume"]:
        fconv = open(os.path.join(args["output"],convergence_name), 'w')
        fconv.write('epoch,split,metric,value\n')
        fconv.close()
    
    # Main training loop
    if args["resume"]:
        epochs = range(ch['epoch']+1,args["nepochs"]+1)
    else:
        epochs = range(args["nepochs"]+1)

    train_step = 0
    num_examples = 0
    for epoch in epochs:
        if args["fast_dev_run"] and epoch - epochs[0] >= 2:
            print("Fast dev run activated - exiting after 2 epochs")
            break

        if args["optimizer"] == 'sgd':
            utils.adjust_learning_rate(optimizer, epoch, args["lr_anneal"], args["lr"])

        ## Training logic
        if epoch == 0 and args["log_example_images"]:
            logging_utils_3D.log_train_example_images_3D(wandb_run, train_dset, n=25)
        if epoch > 0:
            loss = train(epoch, train_loader, model, criterion, optimizer, train_step, num_examples)
        else:
            loss = np.nan
        ### Printing stats
        fconv = open(os.path.join(args["output"],convergence_name), 'a')
        fconv.write('{},train,loss,{}\n'.format(epoch, loss))
        fconv.close()
        
        ### Validation logic
        # Evaluate on train data
        train_probs, train_ground_truth, train_loss = test(epoch, train_for_evaluation_loader, model, criterion)
        train_results_dict = calculate_metrics_from_probs(train_probs, train_ground_truth)
        train_results_dict['loss'] = train_loss
        train_results_dict['num_examples'] = num_examples
        log_metrics(wandb_run, train_results_dict, epoch, train_step, prefix='epoch/train/')

        # Evaluate on validation set
        val_probs, val_ground_truth, val_loss = test(epoch, val_loader, model, criterion)
        val_results_dict = calculate_metrics_from_probs(val_probs, val_ground_truth)
        val_results_dict['loss'] = val_loss
        log_metrics(wandb_run, val_results_dict, epoch, train_step, prefix='epoch/val/')
        
        # log some other stuff
        if args['log_example_images']:
            logging_utils_3D.log_val_example_images_3D(wandb_run, val_dset, val_probs, val_ground_truth, model, n_table_ex=50, n=5)
        log_classification_plots(wandb_run, val_probs, val_ground_truth)
        log_probability_histogram(wandb_run, val_probs, val_ground_truth)
        
        ### Create checkpoint dictionary
        obj = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'lr_scheduler' : lr_scheduler.lr_scheduler.state_dict(),
            'early_stopping' : {'best_loss':early_stopping.best_loss,'counter':early_stopping.counter,'early_stop':early_stopping.early_stop,'min_delta': early_stopping.min_delta,'patience': early_stopping.patience},
            'auc': val_results_dict['auc']
        }
        
        ### Save checkpoint
        torch.save(obj, os.path.join(args["output"],'checkpoint_seed'+str(args["seed_index"])+'_run'+str(args["run"])+'.pth'))

        #### Save best checkpoint
        if val_results_dict['auc'] >= best_auc:
            best_auc = val_results_dict['auc']
            torch.save(obj, os.path.join(args["output"],'checkpoint_seed'+str(args["seed_index"])+'_run'+str(args["run"])+'_best.pth'))
        
        ### Early stopping
        if args["lr_scheduler"]:
            lr_scheduler(-val_results_dict['auc'])
        if args["early_stopping"]:
            early_stopping(-val_results_dict['auc'])
            if early_stopping.early_stop:
                break
    wandb_run.finish()
    log_file.close()

def test(epoch, loader, model, criterion):
    # Set model in test mode
    model.eval()
    # Initialize probability vector
    probs = torch.FloatTensor(len(loader.dataset)).cuda()
    ground_truth = torch.zeros(len(loader.dataset), dtype=int).cuda() 

    # Initialize loss
    running_loss = 0.

    # Loop through batches
    with torch.no_grad():
        for i, (input, target) in enumerate(tqdm.tqdm(loader, file=sys.stdout)):
            if args["fast_dev_run"] and i >= 1:
                break

            ## Copy batch to GPU
            input = input.cuda()
            target_1hot = F.one_hot(target.long(),num_classes=2).cuda()

            ## Forward pass
            y = model(input) #features, probabilities

            loss = criterion(y, target_1hot.float())
            running_loss += loss.item()*input.size(0)

            pred = F.softmax(y,dim=1)

            ## Clone output to output vector
            probs[i*args["batch_size"]:i*args["batch_size"]+input.size(0)] = pred.detach()[:,1].clone()
            ground_truth[i*args["batch_size"]:i*args["batch_size"]+input.size(0)] = target.int().cuda()
    
    if args["fast_dev_run"]:
        # just trim to the first batch
        probs = probs[:input.size(0)]
        ground_truth = ground_truth[:input.size(0)]
    
    running_loss /= probs.size(0)
    return probs.cpu().numpy(), ground_truth.cpu().numpy(), running_loss

def train(epoch, loader, model, criterion, optimizer):
    # Set model in training mode
    model.train()
    # Initialize loss
    running_loss = 0.
    # Loop through batches
    for i, (input,target) in enumerate(tqdm.tqdm(loader, file=sys.stdout)):
        if args["fast_dev_run"] and i >= 1:
            break

        ## Copy to GPU
        input = input.cuda()
        target_1hot = F.one_hot(target.long(),num_classes=2).cuda()
        ## Forward pass
        y = model(input) #features, probabilities

        ## Calculate loss
        loss = criterion(y, target_1hot.float())
        ## Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ## Store loss
        running_loss += loss.item()*input.size(0)
        num_examples += input.size(0)

        # log step metrics
        y_true = target.detach().cpu().numpy()
        y_probs = F.softmax(y, dim=1)[:,1].detach().cpu().numpy()

        step_results_dict = calculate_metrics_from_probs(y_probs, y_true)
        step_results_dict['loss'] = loss.item()
        step_results_dict['num_examples'] = num_examples + input.size(0)
        log_metrics(wandb_run, step_results_dict, epoch, train_step, prefix='step/train/')        

        train_step += 1

    return running_loss/len(loader.dataset), train_step, num_examples

if __name__ == '__main__':
    main()
