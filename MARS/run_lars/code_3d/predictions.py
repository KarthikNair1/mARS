import dataset
import utils
import os
import argparse
import numpy as np
import pandas as pd
import torch.backends.cudnn as cudnn
import torch
import torch.nn.functional as F
import time
import sys
import yaml
import shutil
from datetime import datetime 
import glob
import tqdm
# import pdb


from dataset import get_datasets_3d, dataset_3d, dataset_3d_center
from utils import model_prediction, my_model

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
        'ct_mode': None,
        'ct_bone_seg_dilation_erosion_pixels': None,
        'ct_bone_seg_dilation_alone_pixels': None
    }

    for key, value in default_dict.items():
        if key not in args:
            args[key] = value


def main():
    # Get user input
    global args
    args = load_config()          

    model_path_pattern = args['model_path_pattern']
    model_paths = glob.glob(model_path_pattern)
    print(f"Found {len(model_paths)} model files matching the pattern.")

    transform_fn = None # can change this later

    set_defaults_config()

    # save config file
    t0 = time.time()
    config_outname = "config_" + datetime.today().strftime('%Y-%m-%d_%H:%M:%S') + ".yaml"
    save_copy_of_config(config_outname)

    for i, chpnt in enumerate(model_paths): 
        print('Model {}...'.format(i))
        dsettype = args['dataset_type'] #train, test or val

        # Get model
        model = utils.model_prediction(chpnt, ct_mode=args["ct_mode"])
        #if args["ct_mode"] is not None:
        #    model = utils.modify_resnet_to_accomodate_two_channels(model)
        model_dict = model.state_dict()
        model.cuda()
        model.eval()
        cudnn.benchmark = True
        # Set datasets
        dset = get_datasets_3d(dsettype, transform=transform_fn, norm=args["normalize"],
                            balance=False, data_path_img = args["data_path_img"], 
                            data_path = args["data_path_datafiles"],
                            postfilter_fwhm_xy_mm=args["postfilter_fwhm_xy_mm"],
                            normalize_mean=args["normalize_mean"],
                            normalize_std=args["normalize_std"],
                            target_label_name=args["target_label_name"],
                            ct_mode = args["ct_mode"],
                            ct_bone_seg_dilation_erosion_pixels = args["ct_bone_seg_dilation_erosion_pixels"],
                            ct_bone_seg_dilation_alone_pixels = args["ct_bone_seg_dilation_alone_pixels"]
                        )

        print("Printing dataset signature:")
        for xa in range(5):
            print(torch.norm(dset[xa][0]))

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
            input_sample = dset[0][0].unsqueeze(0).cuda()
            #f, output = model.forward_feat(input_sample)
            output = model(input_sample)
            print(f'{output}')
        
        print('  dataset size: {}'.format(len(dset.df)))

        out = predict(dset, model)
        pred_name = f'pred_{args["dataset_type"]}_{str(i)}.csv'
        out.to_csv(os.path.join(args["output"], pred_name), index=False)
        print('  loop time: {:.0f} min'.format((time.time()-t0)/60))
        t0 = time.time()
        
def predict(dset, model):
    # Set loaders
    loader = torch.utils.data.DataLoader(dset, batch_size=args["batch_size"], shuffle=False, num_workers=args["workers"])    
    # Evaluate on test data
    probs, feats = test(loader, model)
    out = dset.df.copy().reset_index(drop=True)
    out['probs'] = probs
    feats = pd.DataFrame(feats, columns=['F{}'.format(x) for x in range(1,args["nfeat"]+1)])
    out = pd.concat([out, feats], axis=1)
    return out

def test(loader, model):
    # Set model in test mode
    model.eval()
    # Initialize probability vector
    probs = torch.FloatTensor(len(loader.dataset)).cuda()
    # Initialize features
    feats = np.empty((len(loader.dataset),args["nfeat"]))
    # Loop through batches
    with torch.no_grad():
        for i, (input,_) in enumerate(tqdm.tqdm(loader)):
            # print('Batch: [{}/{}]'.format(i+1, len(loader)))
            ## Copy batch to GPU
            input = input.cuda()
            ## Forward pass
            f, output = model.forward_feat(input)
            output = F.softmax(output, dim=1)
            ## Features
            feats[i*args["batch_size"]:i*args["batch_size"]+input.size(0),:] = f.detach().cpu().numpy()
            ## Clone output to output vector
            probs[i*args["batch_size"]:i*args["batch_size"]+input.size(0)] = output.detach()[:,1].clone()
    return probs.cpu().numpy(), feats

if __name__ == '__main__':
    main()
