import dataset
import utils
import os
import argparse
import torch.backends.cudnn as cudnn
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
import pdb

parser = argparse.ArgumentParser(description='PET lymphoma prediction')

#I/O PARAMS
parser.add_argument('--output', type=str, default='.', help='name of output folder')
parser.add_argument('--seeds', type=int, default=None, nargs='+', help='seed range to predict')

#MODEL PARAMS
parser.add_argument('--chptfolder', type=str, default='/gpfs/data/mayerhoeferlab/projects/lars/lars-avg-max_chpt/', help='path to trained model')
parser.add_argument('--normalize', action='store_true', default=False, help='normalize images')
parser.add_argument('--nfeat', type=int, default=512, help='number of embedded features')
   
#TRAINING PARAMS
parser.add_argument('--batch_size', type=int, default=20, help='how many images to sample per slide (default: 100)')
parser.add_argument('--workers', default=5, type=int, help='number of data loading workers (default: 10)')

def main():
    # Get user input
    global args
    args = parser.parse_args()   
    #pdb.set_trace()            
    # set per seed
    if args.seeds is not None:
        seeds = range(args.seeds[0],args.seeds[1]+1)
    else:
        seeds = range(10)
    t0 = time.time()
    for seed in seeds:
        print('Seed {}...'.format(seed))
        # Checkpoint of this seed
        chpnt = os.path.join(args.chptfolder,'LARS_trained_state-dict_'+str(seed)+'.pth')
        # Get model
        model = utils.model_prediction(chpnt)
        model.cuda()
        cudnn.benchmark = True
        # Set datasets
        dset = dataset.get_datasets_pf(args.normalize,False)
        print('  dataset size: {}'.format(len(dset.df)))
        # pdb.set_trace()
        out = predict(dset, model)
        pred_name = 'pred_seed'+str(seed)+'.csv'
        out.to_csv(os.path.join(args.output, pred_name), index=False)
        print('  loop time: {:.0f} min'.format((time.time()-t0)/60))
        t0 = time.time()
        
def predict(dset, model):
    # Set loaders
    loader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)    
    # Evaluate on test data
    probs, feats = test(loader, model)
    out = dset.df.copy().reset_index(drop=True)
    out['probs'] = probs
    feats = pd.DataFrame(feats, columns=['F{}'.format(x) for x in range(1,args.nfeat+1)])
    out = pd.concat([out, feats], axis=1)
    return out

def test(loader, model):
    # Set model in test mode
    model.eval()
    # Initialize probability vector
    probs = torch.FloatTensor(len(loader.dataset)).cuda()
    # Initialize features
    feats = np.empty((len(loader.dataset),args.nfeat))
    # Loop through batches
    with torch.no_grad():
        for i, (input) in enumerate(loader):
            # print('Batch: [{}/{}]'.format(i+1, len(loader)))
            ## Copy batch to GPU
            input = input.cuda()
            ## Forward pass
            f, output = model.forward_feat(input)
            output = F.softmax(output, dim=1)
            ## Features
            feats[i*args.batch_size:i*args.batch_size+input.size(0),:] = f.detach().cpu().numpy()
            ## Clone output to output vector
            probs[i*args.batch_size:i*args.batch_size+input.size(0)] = output.detach()[:,1].clone()
    return probs.cpu().numpy(), feats

if __name__ == '__main__':
    main()
