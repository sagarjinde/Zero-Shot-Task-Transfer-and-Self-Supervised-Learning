# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import re
import PIL
from PIL import Image
# %matplotlib inline
import skimage
from skimage.transform import resize

num_workers = 0
batch_size = 32
test_size = 0.02
num_models = 48
epochs = 100

def resize_image(im, new_dims):
    """
    Resize an image array with interpolation.
    Parameters
    ----------
    im : (H x W x K) ndarray
    new_dims : (height, width) tuple of new dimensions.
    interp_order : interpolation order, default is linear.
    Returns
    -------
    im : resized ndarray with shape (new_dims[0], new_dims[1], K)
    """
    return resize(im, new_dims, order=1, preserve_range=True)

def rescale_image(im, current_scale, new_scale=[-1.,1.], no_clip=False):
    """
    Rescales an image pixel values to target_scale
    
    Args:
        img: A np.float_32 array, assumed between [0,1]
        new_scale: [min,max] 
        current_scale: If not supplied, it is assumed to be in:
            [0, 1]: if dtype=float
            [0, 2^16]: if dtype=uint
            [0, 255]: if dtype=ubyte
    Returns:
        rescaled_image
    """
    im = skimage.img_as_float(im).astype(np.float32)
    if current_scale is not None:
        min_val, max_val = current_scale
        if not no_clip:
            im = np.clip(im, min_val, max_val)
        im = im - min_val
        im /= (max_val - min_val) 
    min_val, max_val = new_scale
    im *= (max_val - min_val)
    im += min_val
    return im 

def resize_rescale_image(img, new_dims, current_scale):
    """
    Resize an image array with interpolation, and rescale to be 
      between 
    Parameters
    ----------
    im : (H x W x K) ndarray
    new_dims : (height, width) tuple of new dimensions.
    new_scale : (min, max) tuple of new scale.
    interp_order : interpolation order, default is linear.
    Returns
    -------
    im : resized ndarray with shape (new_dims[0], new_dims[1], K)
    """
    img = skimage.img_as_float(img)
    img = resize_image(img, new_dims=new_dims)
    img = rescale_image(img, current_scale=current_scale)
    return img

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, y_path, y_filenames):
        'Initialization'
        self.y_path = y_path
        self.y_filenames = y_filenames

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.y_filenames)

  def __getitem__(self, index):
        'Generates one sample of data'
        Z = Image.open(os.path.join(self.y_path, self.y_filenames[index]))
        z = transforms.ToTensor()(Z)
        y = resize_rescale_image(z, new_dims=(1,256,256), current_scale=[0.0,0.08])		# resize-rescale image
        y = torch.from_numpy(y)
        x = resize_rescale_image(z, new_dims=(3,256,256), current_scale=None)		# resize-rescale image
        x = torch.from_numpy(x)
        return x, y

path = os.path.dirname(os.path.realpath(__file__))
parent_path = path+'/taskonomy-sample-model-1'
input_path = path+'/input_model'
y_dir = 'rgb'

y_path = os.path.join(parent_path, y_dir)
y_filenames = os.listdir(y_path)
y_filenames.sort(key=lambda f: int(re.sub('\D', '', f)))

num_train = len(y_filenames)
indices = list(range(num_train))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3,5,kernel_size=3),
            nn.ReLU(True),
            nn.Conv2d(5,8,kernel_size=3),
            nn.ReLU(True))

        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(8,5,kernel_size=3),
            nn.ReLU(True),
            nn.ConvTranspose2d(5,1,kernel_size=3),
            nn.ReLU(True))
    
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

for p in range(1, num_models+1): 
   
    model = Net()
    model.cuda()

    # loss function (cross entropy loss)
    criterion = nn.MSELoss()

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr = 0.001)

    # tracks validation loss change after each epoch
    minimum_validation_loss = np.inf 

    print('==> model number: ',p)
    #Randomize indices
    np.random.shuffle(indices)
    split = int(np.floor(num_train*test_size))
    train_index, test_index = indices[split:], indices[:split]# Making samplers for training and validation batches
    train_index = train_index[:1000]        # taking first 1000

    train_sampler = SubsetRandomSampler(train_index)
    test_sampler = SubsetRandomSampler(test_index)# Creating data loaders
    
    dataset = Dataset(y_path, y_filenames)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers)

    for epoch in range(1, epochs+1):

        train_loss = 0
        valid_loss = 0
    
        # training steps
        model.train()
        for batch_index, (data, target) in enumerate(train_loader):
            # moves tensors to GPU
            data, target = data.cuda(), target.cuda()
            # clears gradients
            optimizer.zero_grad()
            # forward pass
            output = model(data)
            # loss in batch
            loss = criterion(output, target)
            # backward pass for loss gradient
            loss.backward()
            # update paremeters
            optimizer.step()
            # update training loss
            train_loss += loss.item()*data.size(0)
            
        # validation steps
        model.eval()
        for batch_index, (data, target) in enumerate(test_loader):
            # moves tensors to GPU
            data, target = data.cuda(), target.cuda()
            # forward pass
            output = model(data)
            # loss in batch
            loss = criterion(output, target)
            # update validation loss
            valid_loss += loss.item()*data.size(0)
            
        # average loss calculations
        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(test_loader.sampler)
        
        # Display loss statistics
        print(f'Current Epoch: {epoch}\nTraining Loss: {round(train_loss, 6)}\nValidation Loss: {round(valid_loss, 6)}')
    
    model_path = input_path+'/edge2d/edge2d_model_'+str(p)+'.pt'
    torch.save(model.state_dict(), model_path)

