import dataset
import utils
import logging_utils

from utils import EarlyStopping, LRScheduler
import os
import pandas as pd
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import time
import tqdm
import shutil
import yaml
import sys
from datetime import datetime 
import time
import random
import wandb
from sklearn.metrics import roc_auc_score


# config functions
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
        os.makedirs(args["output"], exist_ok=True)

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
        'postfilter_fwhm_xy_mm_max': None,
        'skip_eval_on_training': True,
        'skip_eval_on_validation': False,
        'suv_clip_threshold_min': None,
        'suv_clip_threshold_max': None,
        'combine_mipviews': None,
        'feature_concat_divide_weights': True,
        'freeze_image_encoder': False,
        'lr_fc': None,
        'lr_scheduler': 'step',
        'lr_scheduler_min_lr': None,
        'lr_scheduler_cycle_length': None,
        'lr_scheduler_warm_restarts_multiplier': None,
        'lr_scheduler_patience': 2,
        'lr_scheduler_reduction_factor': 0.1,
        'ct_mode': None,
        'ct_bone_seg_dilation_erosion': None,
        'ct_bone_seg_dilation_alone': None,
        'train_test_split_pkl': None,
        'run_eval_every_n_epochs': None,
        'eval_on_fraction_of_val': None,
        'auxiliary_head_label_names': [],
        'auxiliary_loss_scale': 1.0
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

def parse_targets(target):
    if isinstance(target, dict):
        return target['main'], target['aux']
    return target, None

def format_head_metrics_for_logging(head_metrics):
    out = {}
    for head_name, metrics in head_metrics.items():
        safe_head_name = str(head_name).replace(' ', '_')
        out[f'head_{safe_head_name}_loss'] = metrics.get('loss', float('nan'))
        out[f'head_{safe_head_name}_auc'] = metrics.get('auc', float('nan'))
    return out

model_names = ['densenet121', 'densenet161', 'densenet169', 'densenet201', 
               'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 
               'shufflenet.v2.x0.5', 'shufflenet.v2.x1.0', 'shufflenet.v2.x1.5', 'shufflenet.v2.x2.0', 
               'squeezenet1.0', 'squeezenet1.1',
               'mobilenet.v2','googlenet','inception.v3']

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
    if '--train_test_split_pkl' in sys.argv:
        train_test_split_pkl_pos = sys.argv.index('--train_test_split_pkl') + 1
        args['train_test_split_pkl'] = sys.argv[train_test_split_pkl_pos]
    if '--output' in sys.argv:
        output_pos = sys.argv.index('--output') + 1
        args['output'] = sys.argv[output_pos]
    if '--skip_eval_on_validation' in sys.argv:
        args['skip_eval_on_validation'] = True
    if '--lr_scheduler_cycle_length' in sys.argv:
        cycle_pos = sys.argv.index('--lr_scheduler_cycle_length') + 1
        args['lr_scheduler_cycle_length'] = int(sys.argv[cycle_pos])
    if '--run_eval_every_n_epochs' in sys.argv:
        run_eval_every_n_epochs_pos = sys.argv.index('--run_eval_every_n_epochs') + 1
        args['run_eval_every_n_epochs'] = int(sys.argv[run_eval_every_n_epochs_pos])
    if '--eval_on_fraction_of_val' in sys.argv:
        eval_on_fraction_of_val_pos = sys.argv.index('--eval_on_fraction_of_val') + 1
        args['eval_on_fraction_of_val'] = sys.argv[eval_on_fraction_of_val_pos]

    set_defaults_config()

    if args['skip_eval_on_validation'] and args.get("lr_scheduler") == 'plateau':
        raise ValueError("lr_scheduler='plateau' requires validation loss; disable --skip_eval_on_validation or choose a different scheduler.")
    if args['skip_eval_on_validation'] and args.get("early_stopping"):
        raise ValueError("early_stopping requires validation loss; disable --skip_eval_on_validation or set early_stopping=false.")
    if args['eval_on_fraction_of_val'] is not None:
        args['eval_on_fraction_of_val'] = float(args['eval_on_fraction_of_val'])
        if not (0 < args['eval_on_fraction_of_val'] <= 1):
            raise ValueError("eval_on_fraction_of_val must be in the range (0, 1].")
    if args['auxiliary_head_label_names'] is None:
        args['auxiliary_head_label_names'] = []
    if isinstance(args['auxiliary_head_label_names'], str):
        args['auxiliary_head_label_names'] = [x.strip() for x in args['auxiliary_head_label_names'].split(',') if x.strip()]
    args['auxiliary_loss_scale'] = float(args['auxiliary_loss_scale'])

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
        model = utils.model_prediction(args["checkpoint"], arch = 'resnet34')
        model.cuda()
        # ch = torch.load(args["checkpoint"])
        # # model.load_state_dict(ch['state_dict'])
        # model_dict = model.state_dict()
        # #pretrained_dict = {k: v for k, v in ch['state_dict'].items() if k in model_dict}
        # #print('Loaded [{}/{}] keys from checkpoint'.format(len(pretrained_dict),len(model_dict)))
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict)
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
    if args['checkpoint'] is None and args['checkpoint_folder'] is None and not args['resume']:
        print("Training from scratch without loading any model weights.")

    if args['combine_mipviews'] is not None:
        model = utils.modify_resnet_to_accomodate_two_mipviews(model, mipview_strategy=args['combine_mipviews'], feature_concat_divide_weights=args['feature_concat_divide_weights'])
        model.cuda()

    if len(args['auxiliary_head_label_names']) > 0:
        print(f"Initializing auxiliary heads: {args['auxiliary_head_label_names']}")
        model.initialize_auxiliary_heads(args['auxiliary_head_label_names'])
        model.cuda()

    model_dict = model.state_dict()

    wandb_run.watch(model, log="all", log_freq=100)
    # Set optimizer
    optimizer = utils.create_optimizer(model, args["optimizer"], args["lr"], args["momentum"], args["wd"], fc_lr = args["lr_fc"])
    if args["resume"] and 'optimizer' in ch:
        optimizer.load_state_dict(ch['optimizer'])
        print('Loaded optimizer state')

    if args['freeze_image_encoder']:
        print("Freezing image encoder weights")
        for param in model.features.parameters():
            param.requires_grad = False
            
    cudnn.benchmark = True
    
    # Set datasets
    flipHorVer = dataset.RandomFlip()
    flipLR     = dataset.RandomFlipLeftRight()
    rot90      = dataset.RandomRot90()
    scale      = dataset.RandomScale()
    noise      = dataset.RandomNoise()
    blur       = dataset.GaussianBlur(kernel_size = args['BLUR_kernel_size'], sigma_low = args['BLUR_sigma_low'], sigma_high = args['BLUR_sigma_high'])

    if args["augm"]==0:
        transform = None
    elif args["augm"]==1:
        transform = transforms.Compose([flipHorVer])
    elif args["augm"]==2: 
        transform = transforms.Compose([rot90])
    elif args["augm"]==3:
        transform = transforms.Compose([flipLR])
    elif args["augm"]==12:
        transform = transforms.Compose([flipHorVer,rot90])
    elif args["augm"]==4:
        transform = transforms.Compose([scale])
    elif args["augm"]==5:
        transform = transforms.Compose([noise])
    elif args["augm"]==6:
        transform = transforms.Compose([blur])
    elif args["augm"]==14:
        transform = transforms.Compose([flipHorVer,scale])
    elif args["augm"]==34:
        transform = transforms.Compose([flipLR,scale])
    elif args["augm"]==45:
        transform = transforms.Compose([scale,noise])

    train_test_split = pd.read_pickle(args['train_test_split_pkl']) if args['train_test_split_pkl'] is not None else None
    
    train_dset = dataset.get_datasets(dataset_type='train', 
        norm=args["normalize"], transform=transform, 
        data_path_img = args["data_path_img"], data_path = args["train_datapath_csv"], 
        subset_ids=train_test_split['train'] if train_test_split is not None else None,
        postfilter_fwhm_xy_mm=args["postfilter_fwhm_xy_mm"],
        normalize_mean=args["normalize_mean"], normalize_std=args["normalize_std"],
        target_label_name=args["target_label_name"],
        postfilter_fwhm_xy_mm_min = args["postfilter_fwhm_xy_mm_min"],
        postfilter_fwhm_xy_mm_max = args["postfilter_fwhm_xy_mm_max"],
        suv_clip_threshold_min = args["suv_clip_threshold_min"],
        suv_clip_threshold_max = args["suv_clip_threshold_max"],
        combine_mipviews = args["combine_mipviews"],
        ct_mode = args["ct_mode"],
        ct_bone_seg_dilation_erosion_pixels = args["ct_bone_seg_dilation_erosion"],
        ct_bone_seg_dilation_alone_pixels = args["ct_bone_seg_dilation_alone"],
        auxiliary_label_names = args["auxiliary_head_label_names"]
    )

    val_dset = dataset.get_datasets(dataset_type='val', 
        norm=args["normalize"], transform=None, 
        data_path_img = args["data_path_img"], data_path = args["val_datapath_csv"],
        subset_ids=train_test_split['val'] if train_test_split is not None else None,
        postfilter_fwhm_xy_mm=args["postfilter_fwhm_xy_mm"],
        normalize_mean=args["normalize_mean"], normalize_std=args["normalize_std"],
        target_label_name=args["target_label_name"],
        postfilter_fwhm_xy_mm_min=None, postfilter_fwhm_xy_mm_max=None,
        combine_mipviews = args["combine_mipviews"],
        ct_mode = args["ct_mode"],
        ct_bone_seg_dilation_erosion_pixels = args["ct_bone_seg_dilation_erosion"],
        ct_bone_seg_dilation_alone_pixels = args["ct_bone_seg_dilation_alone"],
        auxiliary_label_names = args["auxiliary_head_label_names"]
    )
    
    train_dset_for_evaluation = dataset.get_datasets(dataset_type='val', 
        norm=args["normalize"], transform=None, 
        data_path_img = args["data_path_img"], data_path = args["train_datapath_csv"],
        subset_ids=train_test_split['train'] if train_test_split is not None else None,
        postfilter_fwhm_xy_mm=args["postfilter_fwhm_xy_mm"],
        normalize_mean=args["normalize_mean"], normalize_std=args["normalize_std"],
        target_label_name=args["target_label_name"],
        postfilter_fwhm_xy_mm_min=None, postfilter_fwhm_xy_mm_max=None,
        combine_mipviews = args["combine_mipviews"],
        ct_mode = args["ct_mode"],
        ct_bone_seg_dilation_erosion_pixels = args["ct_bone_seg_dilation_erosion"],
        ct_bone_seg_dilation_alone_pixels = args["ct_bone_seg_dilation_alone"],
        auxiliary_label_names = args["auxiliary_head_label_names"]
    )

    if len(args["auxiliary_head_label_names"]) > 0:
        for aux_label_name in args["auxiliary_head_label_names"]:
            if aux_label_name not in train_dset.df.columns:
                print(f"WARNING: Auxiliary label '{aux_label_name}' is not present in training dataframe columns. This head will receive zero loss.")
                continue
            train_non_missing = train_dset.df[aux_label_name].notna().sum()
            val_non_missing = val_dset.df[aux_label_name].notna().sum() if aux_label_name in val_dset.df.columns else 0
            print(
                f"Auxiliary label '{aux_label_name}' availability: "
                f"train {train_non_missing}/{len(train_dset.df)}, "
                f"val {val_non_missing}/{len(val_dset.df)}"
            )

    print("Printing dataset signature (val):")
    for xa in range(5):
        print(torch.norm(val_dset[xa][0]))
    
    print("Printing model signature:")
    norm_sum = 0
    for param in model_dict.values():
        if torch.is_tensor(param):
            norm_sum += torch.sum(param)
        else:
            norm_sum += param
    print(f"  model parameter sum: {norm_sum.item()}")
    print(f'Is in training mode: {model.training}')

    print("Printing model evaluation on dataset sample signature:")
    model.eval()
    with torch.no_grad():
        input_sample = val_dset[0][0].unsqueeze(0).cuda()
        f, output = model.forward_feat(input_sample)
        print(f'{torch.norm(f)} {output}')
    model.train()
    
    print('Datasets train:{}, val:{}'.format(len(train_dset.df),len(val_dset.df))) 
    
    # Set loss criterion
    if args["balance"]:
        raise NotImplementedError("Balance option not implemented.")
        w = torch.Tensor(balance_weight_neg_pos)
        print('Balance loss with weights:',balance_weight_neg_pos)
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

    val_smaller_loader = None
    if args['eval_on_fraction_of_val'] is not None and args['eval_on_fraction_of_val'] < 1:
        n_val_examples = len(val_dset)
        n_eval_examples = max(1, int(np.ceil(n_val_examples * args['eval_on_fraction_of_val'])))
        rng = np.random.default_rng(2025)
        subset_indices = rng.choice(n_val_examples, size=n_eval_examples, replace=False)
        val_smaller_dset = torch.utils.data.Subset(val_dset, subset_indices.tolist())
        val_smaller_loader = torch.utils.data.DataLoader(
            val_smaller_dset,
            batch_size=args["batch_size"],
            shuffle=False,
            num_workers=args["workers"]
        )
        print(
            f"Created fixed val_smaller dataloader with {n_eval_examples}/{n_val_examples} "
            f"samples ({args['eval_on_fraction_of_val']:.3f} of val set)."
        )

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
        else:
            pass  # other lr schedulers are called elsewhere
        

        
        ## Training logic
        if epoch == 0 and args['log_example_images']:
            logging_utils.log_train_example_images(wandb_run, train_dset, n=25)
        if epoch > 0:
            loss, train_step, num_examples, train_distribution_logging = train(epoch, train_loader, model, criterion, optimizer, train_step, num_examples, lr_scheduler=lr_scheduler, epoch_0=epochs[0])
            logging_utils.log_data_distribution(train_distribution_logging['all_values'], wandb_run, prefix='distribution/train/')
            train_distribution_logging = None  # free memory
        else:
            loss = np.nan
        ### Printing stats
        fconv = open(os.path.join(args["output"],convergence_name), 'a')
        fconv.write('{},train,loss,{}\n'.format(epoch, loss))
        fconv.close()
        
        ### Validation logic
        # Evaluate on train data
        if not args['skip_eval_on_training']:
            train_probs, train_ground_truth, train_loss, trainval_distribution_logging, train_head_metrics = test(epoch, train_for_evaluation_loader, model, criterion)
            train_results_dict = logging_utils.calculate_metrics_from_probs(train_probs, train_ground_truth)
            train_results_dict['loss'] = train_loss
            train_results_dict['num_examples'] = num_examples
            train_results_dict.update(format_head_metrics_for_logging(train_head_metrics))
            logging_utils.log_metrics(wandb_run, train_results_dict, epoch, train_step, prefix='epoch/train/')
        elif epoch > 0:
            # log at least the running loss from training
            train_results_dict = {'loss': loss, 'num_examples': num_examples}
            logging_utils.log_metrics(wandb_run, train_results_dict, epoch, train_step, prefix='epoch/train/')
            

        # Evaluate on validation set
        val_eval_metadata = {
            'eval_source': 'skipped',
            'eval_num_examples': 0,
            'eval_fraction_of_full': 0.0,
            'eval_is_fractional': 0
        }
        if args['skip_eval_on_validation']:
            print("Skipping validation evaluation for this epoch (--skip_eval_on_validation enabled).")
            val_results_dict = {'loss': float('nan'), 'auc': float('nan')}
        else:
            run_full_eval_this_epoch = (
                args['run_eval_every_n_epochs'] is not None
                and epoch % args['run_eval_every_n_epochs'] == 0
            )
            run_fractional_eval_this_epoch = val_smaller_loader is not None and not run_full_eval_this_epoch

            if run_fractional_eval_this_epoch:
                val_loader_this_epoch = val_smaller_loader
                print("Running validation using fixed val_smaller dataloader.")
            elif args['run_eval_every_n_epochs'] is not None and val_smaller_loader is None and epoch % args['run_eval_every_n_epochs'] != 0:
                print(f"Skipping validation evaluation for this epoch (only running every {args['run_eval_every_n_epochs']} epochs).")
                val_results_dict = {'loss': float('nan'), 'auc': float('nan')}
                val_loader_this_epoch = None
            else:
                val_loader_this_epoch = val_loader
                if run_full_eval_this_epoch:
                    print(f"Running full validation this epoch due to run_eval_every_n_epochs={args['run_eval_every_n_epochs']}.")

            if val_loader_this_epoch is not None:
                val_probs, val_ground_truth, val_loss, val_distribution_logging, val_head_metrics = test(epoch, val_loader_this_epoch, model, criterion)
                logging_utils.log_data_distribution(val_distribution_logging['all_values'], wandb_run, prefix='distribution/val/')
                val_distribution_logging = None  # free memory

                eval_num_examples = len(val_probs)
                val_eval_metadata['eval_num_examples'] = eval_num_examples
                val_eval_metadata['eval_fraction_of_full'] = eval_num_examples / len(val_dset)
                val_eval_metadata['eval_is_fractional'] = int(run_fractional_eval_this_epoch)
                val_eval_metadata['eval_source'] = 'val_smaller' if run_fractional_eval_this_epoch else 'full_val'

                val_results_dict = logging_utils.calculate_metrics_from_probs(val_probs, val_ground_truth)
                val_results_dict['loss'] = val_loss
                val_results_dict.update(val_eval_metadata)
                val_results_dict.update(format_head_metrics_for_logging(val_head_metrics))
                logging_utils.log_metrics(wandb_run, val_results_dict, epoch, train_step, prefix='epoch/val/')

                # log some other stuff
                if args['log_example_images'] and not run_fractional_eval_this_epoch:
                    logging_utils.log_val_example_images(wandb_run, val_dset, val_probs, val_ground_truth, model, n_table_ex=50, n=5, combine_mipviews = args['combine_mipviews'])
                elif args['log_example_images']:
                    print("Skipping val example image logging for val_smaller evaluation (requires full val indexing).")

                logging_utils.log_classification_plots(wandb_run, val_probs, val_ground_truth)
                logging_utils.log_probability_histogram(wandb_run, val_probs, val_ground_truth)

                if args['log_mipview_aggregation'] and args['combine_mipviews'] is None:
                    if not run_fractional_eval_this_epoch:
                        # do mip-view aggregation to mean or max for additional logging
                        accessions = val_dset.df['accession'].values
                        val_probs_df = pd.DataFrame({'accession': accessions, 'prob': val_probs, 'truth': val_ground_truth})

                        # assert that every accession in val_probs_df has exactly 2 entries
                        assert all(val_probs_df['accession'].value_counts() == 2), "Not all accessions have exactly 2 mipviews"

                        val_probs_mean = val_probs_df.groupby('accession').mean().reset_index()
                        val_probs_max = val_probs_df.groupby('accession').max().reset_index()

                        logging_utils.log_classification_plots(wandb_run, val_probs_mean['prob'].values, val_probs_mean['truth'].values, prefix='mipview/mean_')
                        logging_utils.log_classification_plots(wandb_run, val_probs_max['prob'].values, val_probs_max['truth'].values, prefix='mipview/max_')
                        mipview_mean_val_results_dict = logging_utils.calculate_metrics_from_probs(val_probs_mean['prob'].values, val_probs_mean['truth'].values)
                        mipview_max_val_results_dict = logging_utils.calculate_metrics_from_probs(val_probs_max['prob'].values, val_probs_max['truth'].values)
                        logging_utils.log_metrics(wandb_run, mipview_mean_val_results_dict, epoch, train_step, prefix='mipview/mean_val_')
                        logging_utils.log_metrics(wandb_run, mipview_max_val_results_dict, epoch, train_step, prefix='mipview/max_val_')
                    else:
                        print("Skipping mip-view aggregation logging for val_smaller evaluation (requires full cor/sag pairs).")

        if val_eval_metadata['eval_source'] == 'skipped':
            logging_utils.log_metrics(wandb_run, val_eval_metadata, epoch, train_step, prefix='epoch/val/')

        #print('Epoch: [{}/{}]\tLoss: {:.6f}\tAUC: {:.4f}\t{:.4f}'.format(epoch, args["nepochs"], loss, train_results_dict['auc'], val_results_dict['auc']))
                
        ### Create checkpoint dictionary
        lr_scheduler_state = lr_scheduler.state_dict() if lr_scheduler is not None else None
        early_stopping_state = {'best_loss':early_stopping.best_loss,'counter':early_stopping.counter,'early_stop':early_stopping.early_stop,'min_delta': early_stopping.min_delta,'patience': early_stopping.patience} if args["early_stopping"] else None
        obj = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'lr_scheduler' : lr_scheduler_state,
            'early_stopping' : early_stopping_state,
            'auc': val_results_dict['auc'],
        }
        ### Save checkpoint
        torch.save(obj, os.path.join(args["output"],'checkpoint_seed'+str(args["seed_index"])+'_run'+str(args["run"])+'.pth'))

        ### Save best checkpoint
        if args['skip_eval_on_validation']:
            # Without validation metrics, keep a best checkpoint alias on every epoch.
            torch.save(obj, os.path.join(args["output"],'checkpoint_seed'+str(args["seed_index"])+'_run'+str(args["run"])+'_best.pth'))
        elif val_results_dict['auc'] >= best_auc:
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
            early_stopping(val_results_dict['loss'])
            if early_stopping.early_stop:
                break

    print("Printing model signature:")
    norm_sum = 0
    for param in model_dict.values():
        if torch.is_tensor(param):
            norm_sum += torch.sum(param)
        else:
            norm_sum += param
    print(f"  model parameter sum: {norm_sum.item()}")

    print("Printing final model evaluation signature on dataset sample signature:")
    
    with torch.no_grad():
        input_sample = val_dset[0][0].unsqueeze(0).cuda()
        f, output = model.forward_feat(input_sample)
        print(f'{torch.norm(f)} {output}')
    
    wandb_run.finish()
    log_file.close()

def test(epoch, loader, model, criterion):
    # Set model in test mode
    model.eval()
    # Initialize probability vector
    probs = torch.zeros(len(loader.dataset), dtype=float).cuda()
    ground_truth = torch.zeros(len(loader.dataset), dtype=int).cuda() 

    # Initialize loss
    running_loss = 0.

    # collect values throughout for logging
    distribution_logging = {'all_values': []}

    # per-head epoch metrics
    head_loss_sums = {'main': 0.0}
    head_loss_counts = {'main': 0}
    head_probs = {'main': []}
    head_truth = {'main': []}

    # Loop through batches
    with torch.no_grad():
        for i, (input, target) in enumerate(tqdm.tqdm(loader, file=sys.stdout)):
            if args["fast_dev_run"] and i >= 1:
                break

            ## Copy batch to GPU
            input = input.cuda()
            num_elements_in_batch = input.size(0)

            if i == 0:
                print("Printing validation data signature:")
                for xx in range(min(5, num_elements_in_batch)):
                    sample = input[xx]
                    print(f'  input norm: {torch.norm(sample)}')
                    print(f'    model output: {model(sample.unsqueeze(0)).detach()}')

            main_target, aux_target = parse_targets(target)

            ## Forward pass
            features, y = model.forward_feat(input)
            aux_logits_dict = model.forward_aux(features) if hasattr(model, 'forward_aux') else {}

            main_loss, _, loss = utils.compute_multitask_loss(y, main_target, aux_logits_dict, aux_target, args['auxiliary_loss_scale'])
            running_loss += loss.item()*num_elements_in_batch

            head_loss_sums['main'] += main_loss.item() * num_elements_in_batch
            head_loss_counts['main'] += num_elements_in_batch

            pred = F.softmax(y,dim=1)
            ## Clone output to output vector
            probs[i*args["batch_size"]:i*args["batch_size"]+num_elements_in_batch] = pred.detach()[:,1].clone()
            ground_truth[i*args["batch_size"]:i*args["batch_size"]+num_elements_in_batch] = main_target.int().cuda()
            head_probs['main'].append(pred.detach()[:,1].cpu().numpy())
            head_truth['main'].append(main_target.detach().cpu().numpy())

            if aux_target is not None and len(aux_logits_dict) > 0:
                aux_target_float = aux_target.float().cuda() if torch.is_tensor(aux_target) else torch.tensor(aux_target, dtype=torch.float32, device=y.device)
                for head_idx, (head_name, head_logits) in enumerate(aux_logits_dict.items()):
                    if head_name not in head_loss_sums:
                        head_loss_sums[head_name] = 0.0
                        head_loss_counts[head_name] = 0
                        head_probs[head_name] = []
                        head_truth[head_name] = []

                    this_aux_labels = aux_target_float[:, head_idx]
                    valid_mask = ~torch.isnan(this_aux_labels)
                    if torch.any(valid_mask):
                        valid_labels = this_aux_labels[valid_mask].long()
                        valid_logits = head_logits[valid_mask]
                        valid_targets_1hot = F.one_hot(valid_labels, num_classes=2).float()
                        this_head_loss = F.binary_cross_entropy_with_logits(valid_logits, valid_targets_1hot, reduction='mean')

                        n_valid = int(valid_mask.sum().item())
                        head_loss_sums[head_name] += this_head_loss.item() * n_valid
                        head_loss_counts[head_name] += n_valid

                        valid_probs = F.softmax(valid_logits, dim=1)[:,1]
                        head_probs[head_name].append(valid_probs.detach().cpu().numpy())
                        head_truth[head_name].append(valid_labels.detach().cpu().numpy())

            # collect values for data distribution logging
            if i == 0:
                distribution_logging['all_values'].extend(input.detach().cpu().numpy().flatten().tolist())

    if args["fast_dev_run"]:
        # just trim to the first batch
        probs = probs[:num_elements_in_batch]
        ground_truth = ground_truth[:num_elements_in_batch]

    running_loss /= probs.size(0)

    head_metrics = {}
    for head_name in head_loss_sums.keys():
        if head_loss_counts[head_name] > 0:
            head_loss = head_loss_sums[head_name] / head_loss_counts[head_name]
        else:
            head_loss = float('nan')

        if len(head_probs[head_name]) > 0:
            y_probs = np.concatenate(head_probs[head_name])
            y_true = np.concatenate(head_truth[head_name]).astype(int)
            try:
                head_auc = roc_auc_score(y_true, y_probs)
            except ValueError:
                head_auc = float('nan')
        else:
            head_auc = float('nan')

        head_metrics[head_name] = {'loss': head_loss, 'auc': head_auc}

    return probs.cpu().numpy(), ground_truth.cpu().numpy(), running_loss, distribution_logging, head_metrics

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
        num_elements_in_batch = input.size(0)

        main_target, aux_target = parse_targets(target)
        ## Forward pass
        features, y = model.forward_feat(input)
        aux_logits_dict = model.forward_aux(features) if hasattr(model, 'forward_aux') else {}

        ## Calculate loss
        main_loss, aux_loss, loss = utils.compute_multitask_loss(y, main_target, aux_logits_dict, aux_target, args['auxiliary_loss_scale'])
        ## Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ## Store loss
        running_loss += loss.item()*num_elements_in_batch
        num_examples += num_elements_in_batch

        # log step metrics
        y_true = main_target.detach().cpu().numpy()
        y_probs = F.softmax(y, dim=1)[:,1].detach().cpu().numpy()
        
        step_results_dict = logging_utils.calculate_metrics_from_probs(y_probs, y_true)
        step_results_dict['loss'] = loss.item()
        step_results_dict['main_loss'] = main_loss.item()
        step_results_dict['aux_loss_unscaled'] = aux_loss.item()
        step_results_dict['num_examples'] = num_examples

        lr_dict_step_dict = logging_utils.get_log_learning_rates_step(wandb_run, optimizer, train_step)
        step_results_dict.update(lr_dict_step_dict)        

        logging_utils.log_metrics(wandb_run, step_results_dict, epoch, train_step, prefix='step/train/')  

        # collect values for data distribution logging
        if i==0:
            train_distribution_logging['all_values'].extend(input.detach().cpu().numpy().flatten().tolist())

        # step the LR scheduler
        if args["lr_scheduler"] == 'cosine' or args["lr_scheduler"] == 'cosine_with_warm_restarts':
            lr_scheduler.step(epoch - epoch_0 - 1 + i/len(loader)) # note the extra -1, because 0th epoch has no training

        train_step += 1

    return running_loss/len(loader.dataset), train_step, num_examples, train_distribution_logging

if __name__ == '__main__':
    main()
