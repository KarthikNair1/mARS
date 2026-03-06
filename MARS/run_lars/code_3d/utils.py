import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import scipy.ndimage
try:
    import resnet_3d
except:
    from MARS.run_lars.code_3d import resnet_3d
import pdb
#'densenet121', 'densenet161', 'densenet169', 'densenet201', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'squeezenet1_0', 'squeezenet1_1'

def model_wrapper(model,input):
    _,out = model(input)
    return out

def model_prediction(chpnt, ct_mode=None):
    # arch=18  
    arch=34
    model = my_model(arch)
    if ct_mode is not None and ct_mode != "BONE_SEG_MASK_MULTIPLY_ONLY":
        model = modify_resnet_to_accomodate_two_channels(model)
    
    ch = torch.load(chpnt, map_location='cuda:0')
    model.load_state_dict(ch['state_dict'])
    return model
    
# Function that returns the model
class my_model(nn.Module):
    # def __init__(self,arch,input_W,input_H,input_D,n_seg_classes,resnet_shortcut):
    def __init__(self,depth):
        super(my_model, self).__init__()
        # CNN
        # pdb.set_trace()
        model = resnet_3d.generate_model(depth)
        # model = resnet_3d.resnet34(
                # sample_input_W=input_W,
                # sample_input_H=input_H,
                # sample_input_D=input_D,
                # shortcut_type=resnet_shortcut,
                # num_seg_classes=n_seg_classes
                # )
        conv1 = model._modules['conv1'].weight.detach().clone().mean(dim=1, keepdim=True)
        model._modules['conv1'] = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        model._modules['conv1'].weight.data = conv1
        model.fc = nn.Linear(model.fc.in_features, 2)
        self.features = nn.Sequential(*list(model.children())[0:-1])
        self.fc = list(model.children())[-1]
        self.flat = True
    
    def forward(self, x):
        f = self.features(x)
        if self.flat:
            # f = torch.flatten(f, 1)
            f = f.view(x.size(0), -1)
        o = self.fc(f)
        if not self.flat:
            o = o.view(x.size(0), -1)
        return o
    
    def forward_feat(self, x):
        f = self.features(x)
        if self.flat:
            # f = torch.flatten(f, 1)
            f = f.view(x.size(0), -1)
        o = self.fc(f)
        if not self.flat:
            o = o.view(x.size(0), -1)
        return f,o

def get_model(depth,cuda=True):
    if 'resnet' in depth:
        depth = int(depth.split('resnet')[-1])
    model = my_model(depth)    
    model = model.cuda()
    return model

def modify_resnet_to_accomodate_two_channels(model_a):
    # stacks the channels for the input

    # modify first conv layer to take in 2 channels instead of 1
    newweights = torch.zeros((64,2,7,7,7))
    newweights[:,0,:,:,:] = model_a.features[0].weight[:,0,:,:,:] # original weights
    torch.nn.init.normal_(newweights[:,1,:,:,:], mean=0.0, std=0.01)
    #newweights[:,1,:,:,:] = model_a.features[0].weight[:,0,:,:,:] # consider random initialization here?

    newconv = nn.Conv3d(2, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
    newconv.weight = nn.Parameter(newweights)
    model_a.features[0] = newconv

    return model_a

def count_model_parameters(model):
    print('Number of     total model parameters:',sum(p.numel() for p in model.parameters()))
    print('Number of trainable model parameters:',sum(p.numel() for p in model.parameters() if p.requires_grad))
    # return sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)

# Function that creates the optimizer
def create_optimizer(model, mode, lr, momentum, wd, fc_lr=None):
    if fc_lr is None:
        fc_lr = lr

    if mode == 'sgd':
        param_groups = [
            {'name': 'encoder', 'params': model.features.parameters(), "lr": lr, "momentum": momentum, "weight_decay": wd, "dampening": 0, "nesterov": True},
            {'name': 'fc', 'params': model.fc.parameters(), "lr": fc_lr, "momentum": momentum, "weight_decay": wd, "dampening": 0, "nesterov": True}
        ]
        optimizer = optim.SGD(param_groups)
    elif mode == 'adam':
        param_groups = [
            {'name': 'encoder', 'params': model.features.parameters(), "lr": lr, "weight_decay": wd},
            {'name': 'fc', 'params': model.fc.parameters(), "lr": fc_lr, "weight_decay": wd}
        ]
        optimizer = optim.Adam(param_groups)
    return optimizer

def model_names():
    return sorted(name for name in models.__dict__
                  if name.islower() and not name.startswith("__")
                  and callable(models.__dict__[name]))

# Function to anneal learning rate
def adjust_learning_rate(optimizer, epoch, period, start_lr, start_lr_fc=None):
    """Sets the learning rate to the initial LR decayed by 10 every period epochs"""
    lr_multiplier = (0.1 ** (epoch // period))
    for param_group in optimizer.param_groups:
        if param_group['name']=='encoder' or start_lr_fc is None:
            lr = start_lr * lr_multiplier
        elif param_group['name']=='fc':
            lr = start_lr_fc * lr_multiplier
        else:
            raise ValueError('Unknown param group name')
        print('New lr:', lr)
        param_group['lr'] = lr
    return optimizer

def smart_binary_dilation(image, kernel_size):
    # perform first (kernel_size,1,1) dilation
    # then (1,kernel_size,1) dilation
    # then (1,1,kernel_size) dilation
    kernel1 =  np.ones((kernel_size,1,1))
    image = image.astype(np.uint8)
    image = scipy.ndimage.binary_dilation(image, structure=kernel1).astype(np.uint8)
    kernel2 =  np.ones((1,kernel_size,1))
    image = scipy.ndimage.binary_dilation(image, structure=kernel2).astype(np.uint8)
    kernel3 =  np.ones((1,1,kernel_size))
    image = scipy.ndimage.binary_dilation(image, structure=kernel3).astype(np.uint8)
    return image

def smart_binary_erosion(image, kernel_size):
    # perform first (kernel_size,1,1) erosion
    # then (1,kernel_size,1) erosion
    # then (1,1,kernel_size) erosion
    kernel1 =  np.ones((kernel_size,1,1))
    image = image.astype(np.uint8)
    image = scipy.ndimage.binary_erosion(image, structure=kernel1).astype(np.uint8)
    kernel2 =  np.ones((1,kernel_size,1))
    image = scipy.ndimage.binary_erosion(image, structure=kernel2).astype(np.uint8)
    kernel3 =  np.ones((1,1,kernel_size))
    image = scipy.ndimage.binary_erosion(image, structure=kernel3).astype(np.uint8)
    return image

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0.001):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(self, optimizer, patience=1, min_lr=1e-5, factor=0.1, cooldown=1, threshold=0.001):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.cooldown = cooldown
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                cooldown=self.cooldown,
                verbose=True
            )
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)