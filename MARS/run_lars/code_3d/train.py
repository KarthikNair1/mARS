import dataset
import utils
from utils import EarlyStopping, LRScheduler
from logging_utils_3D import calculate_metrics_from_probs, log_metrics, log_classification_plots, log_probability_histogram, get_log_learning_rates_step
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
        'log_example_images': True,
        'BLUR_kernel_size': 5,
        'BLUR_sigma_low': 0.1,
        'BLUR_sigma_high': 2.0,
        'postfilter_fwhm_xy_mm_min': None,
        'postfilter_fwhm_xy_mm_max': None,
        'skip_eval_on_training': True,
        'fixed_suv_clip': None,
        'suv_clip_threshold_min': None,
        'suv_clip_threshold_max': None,
        'lr_fc': None,
        'ct_mode': None,
        'ct_clip_threshold_low': -125,
        'ct_clip_threshold_high': 225,
        'ct_bone_seg_dilation_erosion_pixels': None,
        'ct_bone_seg_dilation_alone_pixels': None,
        'track_input_gradients': False,
        'lr_scheduler': None,
        'lr_scheduler_min_lr': None,
        'lr_scheduler_cycle_length': None,
        'lr_scheduler_warm_restarts_multiplier': None,
        'lr_scheduler_patience': 2,
        'lr_scheduler_reduction_factor': 0.1
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

    # allow command line overrides for checkpoint and seed_index 
    if '--checkpoint' in sys.argv:
        checkpoint_pos = sys.argv.index('--checkpoint') + 1
        args['checkpoint'] = sys.argv[checkpoint_pos]
    if '--seed_index' in sys.argv:
        seed_index_pos = sys.argv.index('--seed_index') + 1
        args['seed_index'] = int(sys.argv[seed_index_pos])

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
        ch = torch.load(args["checkpoint"], map_location='cuda:0')
        # model.load_state_dict(ch['state_dict'])
        model = utils.model_prediction(args["checkpoint"])
        
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
        ch = torch.load( os.path.join(args["output"],'checkpoint_seed'+str(args["seed_index"])+'_run'+str(args["run"])+'.pth'), map_location='cuda:0')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in ch['state_dict'].items() if k in model_dict}
        print('Loaded [{}/{}] keys from checkpoint'.format(len(pretrained_dict),len(model_dict)))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if args["ct_mode"] is not None and 'ONLY' not in args['ct_mode']:
        model = utils.modify_resnet_to_accomodate_two_channels(model)

    model.cuda()

    # log gradients and parameters of the model
    wandb_run.watch(model, log='all', log_freq = 50)
    
    # Set optimizer
    optimizer = utils.create_optimizer(model, args["optimizer"], args["lr"], args["momentum"], args["wd"], fc_lr = args["lr_fc"])
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
        postfilter_fwhm_xy_mm_max = args["postfilter_fwhm_xy_mm_max"],
        fixed_suv_clip = args["fixed_suv_clip"],
        suv_clip_threshold_min = args["suv_clip_threshold_min"],
        suv_clip_threshold_max = args["suv_clip_threshold_max"],
        ct_mode = args["ct_mode"],
        ct_clip_threshold_low = args["ct_clip_threshold_low"],
        ct_clip_threshold_high = args["ct_clip_threshold_high"],
        ct_bone_seg_dilation_erosion_pixels = args["ct_bone_seg_dilation_erosion_pixels"],
        ct_bone_seg_dilation_alone_pixels = args["ct_bone_seg_dilation_alone_pixels"]
    )
    val_dset = dataset.get_datasets_3d(dataset_type='val', 
        norm=args["normalize"], transform=None, 
        data_path_img = args["data_path_img"], data_path = args["val_datapath_csv"],
        postfilter_fwhm_xy_mm=args["postfilter_fwhm_xy_mm"],
        normalize_mean=args["normalize_mean"], normalize_std=args["normalize_std"],
        target_label_name=args["target_label_name"],
        postfilter_fwhm_xy_mm_min=None, postfilter_fwhm_xy_mm_max=None,
        fixed_suv_clip = args["fixed_suv_clip"],
        ct_mode = args["ct_mode"],
        ct_clip_threshold_low = args["ct_clip_threshold_low"],
        ct_clip_threshold_high = args["ct_clip_threshold_high"],
        ct_bone_seg_dilation_erosion_pixels = args["ct_bone_seg_dilation_erosion_pixels"],
        ct_bone_seg_dilation_alone_pixels = args["ct_bone_seg_dilation_alone_pixels"]
    )
    train_dset_for_evaluation = dataset.get_datasets_3d(dataset_type='val', 
        norm=args["normalize"], transform=None, 
        data_path_img = args["data_path_img"], data_path = args["train_datapath_csv"],
        postfilter_fwhm_xy_mm=args["postfilter_fwhm_xy_mm"],
        normalize_mean=args["normalize_mean"], normalize_std=args["normalize_std"],
        target_label_name=args["target_label_name"],
        postfilter_fwhm_xy_mm_min=None, postfilter_fwhm_xy_mm_max=None,
        fixed_suv_clip = args["fixed_suv_clip"],
        ct_mode = args["ct_mode"],
        ct_clip_threshold_low = args["ct_clip_threshold_low"],
        ct_clip_threshold_high = args["ct_clip_threshold_high"],
        ct_bone_seg_dilation_erosion_pixels = args["ct_bone_seg_dilation_erosion_pixels"],
        ct_bone_seg_dilation_alone_pixels = args["ct_bone_seg_dilation_alone_pixels"]
    )

    print('Datasets train:{}, val:{}'.format(len(train_dset.df),len(val_dset.df))) 
    
    # Set loss criterion
    if args["balance"]:
        raise NotImplementedError("Balance option not implemented.")
        w = torch.Tensor(balance_weight_neg_pos)
        criterion = nn.BCEWithLogitsLoss(pos_weight=w).cuda()
    else:
        criterion = nn.BCEWithLogitsLoss().cuda()
    
    # Early stopping and LR scheduler
    if args["lr_scheduler"] is not None:
        print('INFO: Initializing learning rate scheduler')
        if args["lr_scheduler"] == 'plateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                optimizer,
                mode='min',
                patience=args["lr_scheduler_patience"],
                factor=args["lr_scheduler_reduction_factor"],
                min_lr=float(args["lr_scheduler_min_lr"]),
                cooldown=0,
                verbose=True
            )
        elif args["lr_scheduler"] == 'cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args["nepochs"], eta_min=args["lr_scheduler_min_lr"])
        elif args["lr_scheduler"] == 'cosine_with_warm_restarts':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args["lr_scheduler_cycle_length"], 
                                                                                T_mult=args["lr_scheduler_warm_restarts_multiplier"], 
                                                                                eta_min=args["lr_scheduler_min_lr"]
                                                                                )
        else:
            lr_scheduler = None
        if args["resume"] and 'lr_scheduler' in ch:
            lr_scheduler.load_state_dict(ch['lr_scheduler'])
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

        if args["optimizer"] == 'sgd' and args["lr_scheduler"] == 'step':
            utils.adjust_learning_rate(optimizer, epoch, args["lr_anneal"], args["lr"], start_lr_fc=args["lr_fc"])
        
        logging_utils_3D.log_learning_rates(wandb_run, optimizer, epoch)
        
        ## Training logic
        if epoch == 0 and args["log_example_images"]:
            logging_utils_3D.log_train_example_images_3D(wandb_run, train_dset, n=25)
        if epoch > 0:
            loss, train_step, num_examples, train_distribution_logging = train(epoch, train_loader, model, criterion, optimizer, train_step, num_examples, lr_scheduler=lr_scheduler, epoch_0=epochs[0])
            logging_utils_3D.log_data_distribution(train_distribution_logging['all_values'], wandb_run, prefix='distribution/train/')
            train_distribution_logging = None  # free memory
        else:
            loss = np.nan
        ### Printing stats
        fconv = open(os.path.join(args["output"],convergence_name), 'a')
        fconv.write('{},train,loss,{}\n'.format(epoch, loss))
        fconv.close()
        
        ### Validation logic
        # Evaluate on train data
        if not args["skip_eval_on_training"]:
            train_probs, train_ground_truth, train_loss, trainval_distribution_logging = test(epoch, train_for_evaluation_loader, model, criterion)
            train_results_dict = calculate_metrics_from_probs(train_probs, train_ground_truth)
            train_results_dict['loss'] = train_loss
            train_results_dict['num_examples'] = num_examples
            log_metrics(wandb_run, train_results_dict, epoch, train_step, prefix='epoch/train/')
        elif epoch > 0:
            # log at least the running loss from training
            train_results_dict = {'loss': loss, 'num_examples': num_examples}
            log_metrics(wandb_run, train_results_dict, epoch, train_step, prefix='epoch/train/')

        # Evaluate on validation set
        val_probs, val_ground_truth, val_loss, val_distribution_logging = test(epoch, val_loader, model, criterion)
        logging_utils_3D.log_data_distribution(val_distribution_logging['all_values'], wandb_run, prefix='distribution/val/')
        val_distribution_logging = None  # free memory

        val_results_dict = calculate_metrics_from_probs(val_probs, val_ground_truth)
        val_results_dict['loss'] = val_loss
        log_metrics(wandb_run, val_results_dict, epoch, train_step, prefix='epoch/val/')
        
        # log some other stuff
        if args['log_example_images']:
            logging_utils_3D.log_val_example_images_3D(wandb_run, val_dset, val_probs, val_ground_truth, model, n_table_ex=50, n=5)
        log_classification_plots(wandb_run, val_probs, val_ground_truth)
        log_probability_histogram(wandb_run, val_probs, val_ground_truth)
        
        ### Create checkpoint dictionary
        lr_scheduler_state = lr_scheduler.state_dict() if args["lr_scheduler"] else None
        early_stopping_state = {'best_loss':early_stopping.best_loss,'counter':early_stopping.counter,'early_stop':early_stopping.early_stop,'min_delta': early_stopping.min_delta,'patience': early_stopping.patience} if args["early_stopping"] else None
        obj = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'lr_scheduler' : lr_scheduler_state,
            'early_stopping' : early_stopping_state,
            'auc': val_results_dict['auc']
        }
        
        ### Save checkpoint
        torch.save(obj, os.path.join(args["output"],'checkpoint_seed'+str(args["seed_index"])+'_run'+str(args["run"])+'.pth'))

        #### Save best checkpoint
        if val_results_dict['auc'] >= best_auc:
            best_auc = val_results_dict['auc']
            torch.save(obj, os.path.join(args["output"],'checkpoint_seed'+str(args["seed_index"])+'_run'+str(args["run"])+'_best.pth'))
        
        ### LR Scheduler and Early stopping
        if args["lr_scheduler"]:
            if args["lr_scheduler"] == 'step':
                pass  # step lr scheduler is called elsewhere
            elif args["lr_scheduler"] == 'plateau':
                lr_scheduler.step(val_results_dict['loss'])
            elif args["lr_scheduler"] == 'cosine':
                pass # called within train steps
            elif args["lr_scheduler"] == 'cosine_with_warm_restarts':
                pass # called within train steps
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

    # collect values throughout training for logging
    distribution_logging = {'all_values': []}

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

            # collect values for data distribution logging
            if i == 0:
                distribution_logging['all_values'].extend(input.detach().cpu().numpy().flatten().tolist())
    
    if args["fast_dev_run"]:
        # just trim to the first batch
        probs = probs[:input.size(0)]
        ground_truth = ground_truth[:input.size(0)]
    
    running_loss /= probs.size(0)
    return probs.cpu().numpy(), ground_truth.cpu().numpy(), running_loss, distribution_logging

def train(epoch, loader, model, criterion, optimizer, train_step, num_examples, lr_scheduler=None, epoch_0=0):
    # Set model in training mode
    model.train()
    # Initialize loss
    running_loss = 0.

    # collect values throughout training for logging
    train_distribution_logging = {'all_values': []}

    # Loop through batches
    for i, (input,target) in enumerate(tqdm.tqdm(loader, file=sys.stdout)):
        if args["fast_dev_run"] and i >= 1:
            break

        ## Copy to GPU
        input = input.cuda()
        if args["track_input_gradients"]:
            input.requires_grad = True
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

        lr_dict_step_dict = logging_utils_3D.get_log_learning_rates_step(wandb_run, optimizer, train_step)
        log_metrics(wandb_run, step_results_dict, epoch, train_step, prefix='step/train/')        

        # collect values for data distribution logging
        if i == 0:
            train_distribution_logging['all_values'].extend(input.detach().cpu().numpy().flatten().tolist())
        
        # log gradients w.r.t. input if applicable
        if args["track_input_gradients"]:
            input_grad = input.grad.detach().cpu().numpy()
            # log mean abs gradient for channel 0 and 1 separately
            mean_abs_grad_ch0 = np.mean(np.abs(input_grad[:,0,:,:,:]))
            mean_abs_grad_ch1 = np.mean(np.abs(input_grad[:,1,:,:,:])) if input_grad.shape[1] > 1 else 0.0
            wandb_run.log({'gradients/input_ch0': mean_abs_grad_ch0,
                            'gradients/input_ch1': mean_abs_grad_ch1,
                            'train_step': train_step})
            
        # step the LR scheduler
        if args["lr_scheduler"] == 'cosine' or args["lr_scheduler"] == 'cosine_with_warm_restarts':
            lr_scheduler.step(epoch - epoch_0 - 1 + i/len(loader)) # note the extra -1, because 0th epoch has no training
                            
        train_step += 1

    return running_loss/len(loader.dataset), train_step, num_examples, train_distribution_logging

if __name__ == '__main__':
    main()
