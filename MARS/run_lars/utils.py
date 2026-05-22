import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import pdb
import scipy.ndimage
#'densenet121', 'densenet161', 'densenet169', 'densenet201', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'squeezenet1_0', 'squeezenet1_1'

def model_wrapper(model,input):
    _,out = model(input)
    return out

def model_prediction(chpnt, arch = 'resnet34'):
    model = my_model(arch)
    ch = torch.load(chpnt,weights_only=False)
    #pdb.set_trace()
    state_dict = ch['state_dict'] if 'state_dict' in ch else ch
    model.load_state_dict(state_dict)
    return model
def model_prediction_dual_mipview(chpnt):
    model = my_model_feat_fusion('resnet34', fusion_mode='concat')
    ch = torch.load(chpnt,weights_only=False)
    state_dict = ch['state_dict'] if 'state_dict' in ch else ch
    model.load_state_dict(state_dict)
    return model

# Function that returns the model
class my_model(nn.Module):
    def __init__(self,arch):
        super(my_model, self).__init__()
        # CNN
        arch = arch.replace('.','_')
        model = models.__dict__[arch](weights='ResNet34_Weights.DEFAULT')
        if arch.startswith('alexnet'):
            model.classifier._modules['6'] = nn.Linear(4096, 2)
            self.features = nn.Sequential(*list(model.children())[0:-1])
            self.fc = list(model.children())[-1]
            self.flat = True
        elif arch.startswith('vgg'):
            model.classifier._modules['6'] = nn.Linear(4096, 2)
            self.features = nn.Sequential(*list(model.children())[0:-1])
            self.fc = list(model.children())[-1]
            self.flat = True
        elif arch.startswith('resnet'):
            conv1 = model._modules['conv1'].weight.detach().clone().mean(dim=1, keepdim=True)
            model._modules['conv1'] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model._modules['conv1'].weight.data = conv1
            model.fc = nn.Linear(model.fc.in_features, 2)
            self.features = nn.Sequential(*list(model.children())[0:-1])
            self.fc = list(model.children())[-1]
            self.flat = True
        elif arch.startswith('inception'):
            conv1 = model._modules['Conv2d_1a_3x3'].conv
            conv1data = model._modules['Conv2d_1a_3x3'].conv.weight.detach().clone().mean(dim=1, keepdim=True)
            model._modules['Conv2d_1a_3x3'].conv = nn.Conv2d(1, conv1.out_channels, kernel_size=conv1.kernel_size[0], stride=conv1.stride[0], padding=conv1.padding[0], bias=False)
            model._modules['Conv2d_1a_3x3'].conv.weight.data = conv1data
            model.fc = nn.Linear(model.fc.in_features, 2)
            self.features = nn.Sequential(*list(model.children())[0:-1])
            self.features.add_module('avgpoolout',nn.AdaptiveAvgPool2d(output_size=(1, 1)))
            self.fc = list(model.children())[-1]
            self.flat = False
        elif arch.startswith('densenet'):
            conv1 = model.features[0]
            conv1data = model.features[0].weight.detach().clone().mean(dim=1, keepdim=True)
            model.features[0] = nn.Conv2d(1, conv1.out_channels, kernel_size=conv1.kernel_size[0], stride=conv1.stride[0], padding=conv1.padding[0], bias=False)
            model.features[0].weight.data = conv1data
            model.classifier = nn.Linear(model.classifier.in_features, 2)
            self.features = nn.Sequential(*list(model.children())[0:-1])
            self.features.add_module('avgpoolout',nn.AdaptiveAvgPool2d(output_size=(1, 1)))
            self.fc = list(model.children())[-1]
            self.flat = True
        elif arch.startswith('squeezenet'):
            conv1 = model.features[0]
            conv1data = model.features[0].weight.detach().clone().mean(dim=1, keepdim=True)
            bias1 = model.features[0].bias.detach().clone()
            model.features[0] = nn.Conv2d(1, conv1.out_channels, kernel_size=conv1.kernel_size[0], stride=conv1.stride[0], padding=conv1.padding[0], bias=True)
            model.features[0].weight.data = conv1data
            model.features[0].bias.data = bias1
            model.classifier._modules['1'] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
            self.features = nn.Sequential(*list(model.children())[0:-1])
            # self.features.add_module('avgpoolout',nn.AdaptiveAvgPool2d(output_size=(1, 1)))
            self.fc = list(model.children())[-1]
            self.flat = False
        elif arch.startswith('shufflenet'):
            conv1 = model.conv1[0]
            conv1data = model.conv1[0].weight.detach().clone().mean(dim=1, keepdim=True)
            model.conv1[0] = nn.Conv2d(1, conv1.out_channels, kernel_size=conv1.kernel_size[0], stride=conv1.stride[0], padding=conv1.padding[0], bias=False)
            model.conv1[0].weight.data = conv1data
            model.fc = nn.Linear(model.fc.in_features, 2)
            self.features = nn.Sequential(*list(model.children())[0:-1])
            self.features.add_module('avgpoolout',nn.AdaptiveAvgPool2d(output_size=(1, 1)))
            self.fc = list(model.children())[-1]
            self.flat = True
        elif arch.startswith('googlenet'):
            conv1 = model._modules['conv1'].conv.weight.detach().clone().mean(dim=1, keepdim=True)
            model._modules['conv1'].conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model._modules['conv1'].conv.weight.data = conv1
            model.fc = nn.Linear(model.fc.in_features, 2)
            self.features = nn.Sequential(*list(model.children())[0:-1])
            self.fc = list(model.children())[-1]
            self.flat = True
        elif arch.startswith('mobilenet'):
            conv1 = model.features[0][0]
            conv1data = model.features[0][0].weight.detach().clone().mean(dim=1, keepdim=True)
            model.features[0][0] = nn.Conv2d(1, conv1.out_channels, kernel_size=conv1.kernel_size[0], stride=conv1.stride[0], padding=conv1.padding[0], bias=False)
            model.features[0][0].weight.data = conv1data
            model.classifier._modules['1'] = nn.Linear(model.classifier._modules['1'].in_features, 2)
            self.features = nn.Sequential(*list(model.children())[0:-1])
            self.features.add_module('avgpoolout',nn.AdaptiveAvgPool2d(output_size=(1, 1)))
            self.fc = list(model.children())[-1]
            self.flat = True
        # self.features = nn.Sequential(*list(model.children())[0:-1])
        # self.fc = list(model.children())[-1]
        self.aux_fcs = nn.ModuleDict()

    def initialize_auxiliary_heads(self, auxiliary_head_names, output_dim=2):
        self.aux_fcs = nn.ModuleDict()
        for head_name in auxiliary_head_names:
            aux_head = nn.Linear(self.fc.in_features, output_dim, bias=self.fc.bias is not None)
            if output_dim == 1:
                # Neutral regression start without inheriting the binary classifier's decision boundary.
                nn.init.xavier_uniform_(aux_head.weight)
                if aux_head.bias is not None:
                    nn.init.zeros_(aux_head.bias)
            else:
                rows_to_copy = min(output_dim, self.fc.out_features)
                aux_head.weight.data[:rows_to_copy].copy_(self.fc.weight.data[:rows_to_copy])
                if self.fc.bias is not None:
                    aux_head.bias.data[:rows_to_copy].copy_(self.fc.bias.data[:rows_to_copy])
            self.aux_fcs[head_name] = aux_head

    def forward_aux(self, features):
        return {head_name: head_layer(features) for head_name, head_layer in self.aux_fcs.items()}
    
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

class my_model_feat_fusion(my_model):
    def __init__(self,arch, fusion_mode = 'concat'):
        super(my_model_feat_fusion, self).__init__(arch)
        # instantiate the extra layers
        self.fc = nn.Linear(1024,2)
        self.fusion_mode = fusion_mode
        self.aux_fcs = nn.ModuleDict()
    def forward(self, x):
        x1 = x[:, 0:1, :, :]
        x2 = x[:, 1:2, :, :]
        f1 = self.features(x1)
        f2 = self.features(x2)
        if self.flat:
            # f = torch.flatten(f, 1)
            f1 = f1.view(x1.size(0), -1)
            f2 = f2.view(x2.size(0), -1)

        # fusion
        if self.fusion_mode == 'concat':
            f = torch.cat((f1,f2),dim=1) # shape is now e.g. (1,1024,1,1)
        else:
            raise NotImplementedError
        
        o = self.fc(f)
        if not self.flat:
            o = o.view(x1.size(0), -1)
        return o
    
    def forward_feat(self, x):
        x1 = x[:, 0:1, :, :]
        x2 = x[:, 1:2, :, :]
        f1 = self.features(x1)
        f2 = self.features(x2)
        if self.flat:
            # f = torch.flatten(f, 1)
            f1 = f1.view(x1.size(0), -1)
            f2 = f2.view(x2.size(0), -1)

        # fusion
        if self.fusion_mode == 'concat':
            f = torch.cat((f1,f2),dim=1) # shape is now e.g. (1,1024)
        else:
            raise NotImplementedError
        
        o = self.fc(f)
        if not self.flat:
            o = o.view(x1.size(0), -1)
        return f,o
    
def get_model(arch):
    model = my_model(arch)    
    model = model.cuda()
    return model

def modify_resnet_to_accomodate_two_mipviews(model_a, mipview_strategy, feature_concat_divide_weights = False):
    if mipview_strategy == 'stack':
        # modify first conv layer to take in 2 channels instead of 1
        newweights = torch.zeros((64,2,7,7))
        newweights[:,0,:,:] = model_a.features[0].weight[:,0,:,:]
        newweights[:,1,:,:] = model_a.features[0].weight[:,0,:,:]

        newconv = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        newconv.weight = nn.Parameter(newweights)
        model_a.features[0] = newconv

        return model_a
    elif mipview_strategy == 'feature_concatenation':
        new_model = my_model_feat_fusion('resnet34', fusion_mode = 'concat')
        new_model = new_model.cuda()
        # copy weights from model_a to new_model
        state_dict = model_a.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')} # don't include fc weights and bias
        new_model.load_state_dict(state_dict, strict=False)
        
        # modify fc layer to take in 1024 features instead of 512
        newweights = torch.zeros((2, 1024))

        newweights[0,:512] = model_a.fc.weight[0,:]
        newweights[0,512:] = model_a.fc.weight[0,:]
        newweights[1,:512] = model_a.fc.weight[1,:]
        newweights[1,512:] = model_a.fc.weight[1,:]

        if feature_concat_divide_weights: # may need to divide weights in order to keep the scale of logits similar
            newweights = newweights/2.0

        new_model.fc = nn.Linear(1024, 2, bias=True)
        new_model.fc.weight = nn.Parameter(newweights)
        new_model.fc.bias = nn.Parameter(model_a.fc.bias)

        return new_model

def count_model_parameters(model):
    print('Number of     total model parameters:',sum(p.numel() for p in model.parameters()))
    print('Number of trainable model parameters:',sum(p.numel() for p in model.parameters() if p.requires_grad))
    # return sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)

# Function that creates the optimizer
def create_optimizer(model, mode, lr, momentum, wd, fc_lr=None):
    if fc_lr is None:
        fc_lr = lr

    head_params = list(model.fc.parameters())
    if hasattr(model, 'aux_fcs') and len(model.aux_fcs) > 0:
        head_params += list(model.aux_fcs.parameters())

    if mode == 'sgd':
        param_groups = [
            {'name': 'encoder', 'params': model.features.parameters(), "lr": lr, "momentum": momentum, "weight_decay": wd, "dampening": 0, "nesterov": True},
            {'name': 'fc', 'params': head_params, "lr": fc_lr, "momentum": momentum, "weight_decay": wd, "dampening": 0, "nesterov": True}
        ]
        optimizer = optim.SGD(param_groups)
    elif mode == 'adam':
        param_groups = [
            {'name': 'encoder', 'params': model.features.parameters(), "lr": lr, "weight_decay": wd},
            {'name': 'fc', 'params': head_params, "lr": fc_lr, "weight_decay": wd}
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
        param_group['lr'] = lr


def transform_aux_targets(aux_target, auxiliary_loss_type):
    if auxiliary_loss_type == 'log_mse':
        return torch.log1p(aux_target.clamp_min(0.0))
    return aux_target

def compute_multitask_loss(main_logits, main_target, aux_logits_dict, aux_target, aux_loss_scale, auxiliary_loss_type='bce'):
    main_target_1hot = F.one_hot(main_target.long(), num_classes=2).cuda().float()
    main_loss = F.binary_cross_entropy_with_logits(main_logits, main_target_1hot, reduction='mean')

    aux_loss = torch.tensor(0.0, device=main_logits.device)
    if aux_target is None or len(aux_logits_dict) == 0:
        return main_loss, aux_loss, main_loss

    if not torch.is_tensor(aux_target):
        aux_target = torch.tensor(aux_target)
    aux_target = aux_target.float().cuda()

    for head_idx, head_name in enumerate(aux_logits_dict.keys()):
        this_aux_labels = aux_target[:, head_idx]
        valid_mask = ~torch.isnan(this_aux_labels)
        if torch.any(valid_mask):
            if auxiliary_loss_type == 'log_mse':
                this_targets = torch.log1p(this_aux_labels[valid_mask].clamp_min(0.0))
                this_predictions = aux_logits_dict[head_name][valid_mask].view(-1)
                this_loss = F.mse_loss(this_predictions, this_targets, reduction='mean')
            elif auxiliary_loss_type == 'mse':
                this_targets = this_aux_labels[valid_mask].float()
                this_predictions = aux_logits_dict[head_name][valid_mask].view(-1)
                this_loss = F.mse_loss(this_predictions, this_targets, reduction='mean')
            elif auxiliary_loss_type == 'bce':
                this_targets_1hot = F.one_hot(this_aux_labels[valid_mask].long(), num_classes=2).to(main_logits.device).float()
                this_loss = F.binary_cross_entropy_with_logits(aux_logits_dict[head_name][valid_mask], this_targets_1hot, reduction='mean')
            else:
                raise ValueError(f"Unknown auxiliary_loss_type: {auxiliary_loss_type}")
            aux_loss = aux_loss + this_loss

    total_loss = main_loss + aux_loss_scale * aux_loss
    return main_loss, aux_loss, total_loss

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