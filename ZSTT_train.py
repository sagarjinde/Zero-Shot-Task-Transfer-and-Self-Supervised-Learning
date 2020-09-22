# INPUT TO BRANCH IS ENCODER ONLY, BUT ARCHITECTURE IS UP OF ALL FC LAYERS
# DROPOUT(0.2) + RELU

import os
import re
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim

import PIL
from PIL import Image
# %matplotlib inline
import skimage
from skimage.transform import resize
"""
from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass
"""
import warnings
warnings.simplefilter("ignore", UserWarning)

# variables for TTNet
TEST_SIZE = 0.2
NUM_TASKS = 3        
OUT_CHANNELS = 4
ENCODER_DIMENSION = 508
DECODER_DIMENSION = 503
INPUT_DIM = ENCODER_DIMENSION + 1
HIDDEN_DIM = 256*3
OUTPUT_DIM = 256
TT_HIDDEN_DIM = 1024
TT_OUTPUT_DIM = 512
NUM_WORKERS = 0
BATCH_SIZE = 16
EPOCHS = 25

# variables for task specific Net
num_workers = 0
batch_size = 32

def resize_image(im, new_dims=(3, 256, 256)):
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

def rescale_image(im, new_scale=[-1.,1.], current_scale=None, no_clip=False):
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

def resize_rescale_image(img):
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
    img = resize_image(img)
    img = rescale_image(img)
    return img

def random_noise_image(img):
    """
        Add noise to an image
        Args:
        im : (H x W x K) ndarray
        new_dims : (height, width) tuple of new dimensions.
        new_scale : (min, max) tuple of new scale.
        interp_order : interpolation order, default is linear.
    Returns:
            a noisy version of the original clean image
    """
    img = skimage.util.img_as_float(img)
    img = resize_image(img)
    img = skimage.util.random_noise(img, var=0.01)
    img = rescale_image(img)
    return img

def permute4x4(images):
    K,H,W = images.size()
    new_dims = (K,H,W)
    unitH = int(H / 4)
    unitW = int(W / 4)

    p_images = torch.FloatTensor(images.size())
    input_imgs = np.empty_like(p_images, dtype=np.float32) 

    target = torch.randperm(16)
    for j in range(16):
        pos = target[j]
        posH = int(pos / 4) * unitH
        posW = int(pos % 4) * unitW        
        i_posH = int(j / 4) * unitH
        i_posW = int(j % 4) * unitW
        p_images[:, i_posH:i_posH+unitH, i_posW:i_posW+unitW] = images[:, posH:posH+unitH, posW:posW+unitW]

    return p_images

class Dataset1(torch.utils.data.Dataset):
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
        Y = Image.open(os.path.join(self.y_path, self.y_filenames[index]))
        Y = Image.open(os.path.join(self.y_path, self.y_filenames[index]))
        Y = transforms.ToTensor()(Y)
        Y = skimage.img_as_float(Y)
        Y = resize_image(Y)
        Y = torch.from_numpy(Y)

        Y = transforms.ToPILImage()(Y).convert("RGB")
        y = transforms.ToTensor()(Y)
        X = permute4x4(y)
        return X, y

class Dataset2(torch.utils.data.Dataset):
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
        z = resize_rescale_image(z)		# resize-rescale image
        z = torch.from_numpy(z)
        y = x = z                               # same image because we are using autoencoder
        return x, y

class Dataset3(torch.utils.data.Dataset):
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
        y = resize_rescale_image(z)             # resize image
        y = torch.from_numpy(y)
        x = random_noise_image(z)		# random noise
        x = torch.from_numpy(x)
        return x, y

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
            nn.ConvTranspose2d(5,3,kernel_size=3),
            nn.ReLU(True))
    
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, x1_path, x1_filenames, x2_path, x2_filenames, x3_path, x3_filenames, gt_path):
        'Initialization'
        self.x1_path = x1_path
        self.x1_filenames = x1_filenames
        self.x2_path = x2_path
        self.x2_filenames = x2_filenames
        self.x3_path = x3_path
        self.x3_filenames = x3_filenames
        self.gt_path = gt_path

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.x1_filenames)

    def __getitem__(self, index):
        'Generates one sample of data'
        X1 = os.path.join(self.x1_path, self.x1_filenames[index])
        X2 = os.path.join(self.x2_path, self.x2_filenames[index])
        X3 = os.path.join(self.x3_path, self.x3_filenames[index])
        # create a model to store data
        gt_model = Net()
        # load weights and extraxt them
        # task-1
        gt_model.load_state_dict(torch.load(X1))
        wgt_task = nn.utils.parameters_to_vector(gt_model.parameters())
        encoder_wgt_task1 = wgt_task[:ENCODER_DIMENSION]
        
        # task-2
        gt_model.load_state_dict(torch.load(X2))
        wgt_task = nn.utils.parameters_to_vector(gt_model.parameters())
        encoder_wgt_task2 = wgt_task[:ENCODER_DIMENSION]
        
        # task-3
        gt_model.load_state_dict(torch.load(X3))
        wgt_task = nn.utils.parameters_to_vector(gt_model.parameters())
        encoder_wgt_task3 = wgt_task[:ENCODER_DIMENSION]

        # ground truth task
        gt_model.load_state_dict(torch.load(self.gt_path))
        wgt_task = nn.utils.parameters_to_vector(gt_model.parameters())
        encoder_gt = wgt_task[:ENCODER_DIMENSION]
        encoder_gt = torch.reshape(encoder_gt, (1,-1))
        decoder_gt = wgt_task[ENCODER_DIMENSION:]
        decoder_gt = torch.reshape(decoder_gt, (1,-1))

        return [encoder_wgt_task1, encoder_wgt_task2, encoder_wgt_task3], [encoder_gt, decoder_gt]

class Branch(nn.Module):
    def __init__(self):
        super(Branch, self).__init__()

        self.initial_fc = nn.Sequential(
                nn.Linear(INPUT_DIM,HIDDEN_DIM),
                nn.Dropout(p=0.2),
                nn.ReLU(),
        )
        self.final_fc = nn.Sequential(
                nn.Linear(HIDDEN_DIM,OUTPUT_DIM),
                nn.Dropout(p=0.2),
                nn.ReLU(),
        )
        
    def forward(self, x):
        x = self.initial_fc(x)
        x = self.final_fc(x)
        return x

class TaskTransferNet(nn.Module):
    def __init__(self, branch1, branch2, branch3):
        super(TaskTransferNet, self).__init__()
        self.branch1 = branch1
        self.branch2 = branch2
        self.branch3 = branch3

        self.initial_fc = nn.Sequential(
                nn.Linear(OUTPUT_DIM*NUM_TASKS,TT_HIDDEN_DIM),
                nn.Dropout(p=0.2),
                nn.ReLU(),
        )
        self.final_fc = nn.Sequential(
                nn.Linear(TT_HIDDEN_DIM,TT_OUTPUT_DIM),
                nn.Dropout(p=0.2),
                nn.ReLU(),
        )

        self.predicted_enc = nn.Sequential(
                nn.Linear(TT_OUTPUT_DIM,ENCODER_DIMENSION),
        )
        self.predicted_dec = nn.Sequential(
                nn.Linear(TT_OUTPUT_DIM,DECODER_DIMENSION),
        )
           
    def forward(self, weights1, weights2, weights3):
        out1 = self.branch1(weights1)
        out2 = self.branch2(weights2)
        out3 = self.branch3(weights3)

        out = torch.cat((out1, out2, out3), dim=0)
        out = self.initial_fc(out)
        out = self.final_fc(out)       
        predicted_enc = self.predicted_enc(out)   
        predicted_dec = self.predicted_dec(out)

        return predicted_enc, predicted_dec

def append_gamma(gamma, wgt_vector):
    gamma = torch.tensor([gamma], dtype=torch.float32)
    return torch.cat((wgt_vector,gamma),dim=0)

def consistency_loss(predicted_weights, data_loader):
    # Load encoder-decoder framework you used to train source tasks
    model = Net()

    # Load predicted encoder-decoder weights in the model
    nn.utils.vector_to_parameters(predicted_weights,model.parameters())

    # Use same criterion you used in training source task
    criterion = torch.nn.MSELoss()

    # Loop over validation data in batches and compute the loss
    total_loss = 0
    model.eval()
    for batch_index, (data, target) in enumerate(data_loader):
        # moves tensors to GPU
        data, target = data.cuda(), target.cuda()
        # forward pass
        output = model(data)
        # loss in batch
        loss = criterion(output, target)
        # update validation loss
        total_loss += loss.item()*data.size(0)
            
    # average loss calculations
    total_loss = total_loss/len(data_loader.sampler)
    return total_loss

def train():

    path = os.path.dirname(os.path.realpath(__file__))

    # images as input to consistency loss
    y_path = path+'/taskonomy-sample-model-1/rgb'
    y_filenames = os.listdir(y_path)
    y_filenames.sort(key=lambda f: int(re.sub('\D', '', f)))

    num_data = len(y_filenames)
    data_index = list(range(num_data))

    # weights as input to TTNet
    x1_path = path + '/input_model/jigsaw'
    x1_filenames = os.listdir(x1_path)
    x1_filenames.sort(key=lambda f: int(re.sub('\D', '', f)))

    x2_path = path + '/input_model/autoencoding'
    x2_filenames = os.listdir(x2_path)
    x2_filenames.sort(key=lambda f: int(re.sub('\D', '', f)))

    x3_path = path + '/input_model/denoising'
    x3_filenames = os.listdir(x3_path)
    x3_filenames.sort(key=lambda f: int(re.sub('\D', '', f)))

    num_train = len(x1_filenames)
    indices = list(range(num_train))

    branch1 = Branch(); branch1.cuda()
    branch2 = Branch(); branch2.cuda()
    branch3 = Branch(); branch3.cuda()

    net = TaskTransferNet(branch1, branch2, branch3)
    net.cuda()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

    ### TASK 1 ###
    print('===> Target: Task-1')
    # data loader for images
    np.random.shuffle(data_index)
    data_index_ = data_index[:200]        # taking first 200

    data_sampler = SubsetRandomSampler(data_index_)
    
    dataset1 = Dataset1(y_path, y_filenames)
    data_loader1 = torch.utils.data.DataLoader(dataset1, batch_size=batch_size, sampler=data_sampler, num_workers=num_workers)

    gt_path = path + '/saved_model/jigsaw/jigsaw_model_27.pt'

    # data loader for weights of task-1
    # Randomize indices
    np.random.shuffle(indices)
    split = int(np.floor(num_train*TEST_SIZE))
    train_index, test_index = indices[split:], indices[:split]# Making samplers for training and validation batches

    train_sampler = SubsetRandomSampler(train_index)
    test_sampler = SubsetRandomSampler(test_index)         

    # Creating data loaders
    dataset = Dataset(x1_path, x1_filenames, x2_path, x2_filenames, x3_path, x3_filenames, gt_path)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler, pin_memory=False, num_workers=NUM_WORKERS)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=test_sampler, pin_memory=False, num_workers=NUM_WORKERS)

    for epoch in range(1, EPOCHS+1):
        print('=> Epoch: ',epoch)
        net.train()
        for batch_index, (data, target) in enumerate(train_loader):
            (encoder_wgt_task1, encoder_wgt_task2, encoder_wgt_task3) = data
            (gt_encoder_wgt, gt_decoder_wgt) = target
            (gt_encoder_wgt, gt_decoder_wgt) = (gt_encoder_wgt.cuda(), gt_decoder_wgt.cuda())

            ################ Self-mode ################
            print('Training in Self-mode')
            net.branch1.require_grad = True
            net.branch2.require_grad = False
            net.branch3.require_grad = False

            loss_batch = 0
            cur_batch_size = encoder_wgt_task1.size()[0]

            for i in range(cur_batch_size):
                # Use append_gamma to append gama
                encoder_wgt_task1_ = append_gamma(5,encoder_wgt_task1[i])
                encoder_wgt_task1_ = encoder_wgt_task1_.cuda()
                encoder_wgt_task2_ = append_gamma(2,encoder_wgt_task2[i])
                encoder_wgt_task2_ = encoder_wgt_task2_.cuda()
                encoder_wgt_task3_ = append_gamma(3,encoder_wgt_task3[i])
                encoder_wgt_task3_ = encoder_wgt_task3_.cuda()

                # forward
                predicted_encoder, predicted_decoder = net(encoder_wgt_task1_, encoder_wgt_task2_, encoder_wgt_task3_)
                predicted_weights = torch.cat((predicted_encoder, predicted_decoder), dim=0)
                predicted_weights = predicted_weights.cuda()

                loss = criterion(predicted_encoder, gt_encoder_wgt) + criterion(predicted_decoder, gt_decoder_wgt) + consistency_loss(predicted_weights, data_loader1)       # introduce lambda
                print('batch_index: {}, i: {}/{}, loss: {}'.format(batch_index,i+1,cur_batch_size,loss.item()))
                loss_batch += loss

            # clears old gradients from the last step
            optimizer.zero_grad()
            loss_batch = loss_batch/cur_batch_size
            loss_batch.backward()     
            print('Self-mode Training loss: ',loss_batch.item())
            optimizer.step()

            ################ Transfer-mode ################
            print('Training in Transfer-mode')
            net.branch1.require_grad = False
            net.branch2.require_grad = True
            net.branch3.require_grad = True

            loss_batch = 0
            cur_batch_size = encoder_wgt_task1.size()[0]

            for i in range(cur_batch_size):
                # Use append_gamma to append gama
                encoder_wgt_task1_ = append_gamma(5,encoder_wgt_task1[i])
                encoder_wgt_task1_ = encoder_wgt_task1_.cuda()
                encoder_wgt_task2_ = append_gamma(2,encoder_wgt_task2[i])
                encoder_wgt_task2_ = encoder_wgt_task2_.cuda()
                encoder_wgt_task3_ = append_gamma(3,encoder_wgt_task3[i])
                encoder_wgt_task3_ = encoder_wgt_task3_.cuda()

                # forward
                predicted_encoder, predicted_decoder = net(encoder_wgt_task1_, encoder_wgt_task2_, encoder_wgt_task3_)
                predicted_weights = torch.cat((predicted_encoder, predicted_decoder), dim=0)
                predicted_weights = predicted_weights.cuda()

                loss = criterion(predicted_encoder, gt_encoder_wgt) + criterion(predicted_decoder, gt_decoder_wgt) + consistency_loss(predicted_weights, data_loader1)       # introduce lambda
                print('batch_index: {}, i: {}/{}, loss: {}'.format(batch_index,i+1,cur_batch_size,loss.item()))
                loss_batch += loss

            # clears old gradients from the last step
            optimizer.zero_grad()
            loss_batch = loss_batch/cur_batch_size
            loss_batch.backward()   
            print('Transfer-mode Training loss: ',loss_batch.item())
            optimizer.step()

        # save model
        tt_model_path = path+'/tt_model/tt_model_1_'+str(epoch)+'.pt'
        torch.save(net.state_dict(), tt_model_path)

        net.eval()
        val_loss = 0
        print('Testing')
        for batch_index, (data, target) in enumerate(test_loader):
            (encoder_wgt_task1, encoder_wgt_task2, encoder_wgt_task3) = data
            (gt_encoder_wgt, gt_decoder_wgt) = target
            (gt_encoder_wgt, gt_decoder_wgt) = (gt_encoder_wgt.cuda(), gt_decoder_wgt.cuda())

            cur_batch_size = encoder_wgt_task1.size()[0]

            for i in range(cur_batch_size):
                # Use append_gamma to append gama
                encoder_wgt_task1_ = append_gamma(5,encoder_wgt_task1[i])
                encoder_wgt_task1_ = encoder_wgt_task1_.cuda()
                encoder_wgt_task2_ = append_gamma(2,encoder_wgt_task2[i])
                encoder_wgt_task2_ = encoder_wgt_task2_.cuda()
                encoder_wgt_task3_ = append_gamma(3,encoder_wgt_task3[i])
                encoder_wgt_task3_ = encoder_wgt_task3_.cuda()

                # forward
                predicted_encoder, predicted_decoder = net(encoder_wgt_task1_, encoder_wgt_task2_, encoder_wgt_task3_)
                predicted_weights = torch.cat((predicted_encoder, predicted_decoder), dim=0)
                predicted_weights = predicted_weights.cuda()

                loss = criterion(predicted_encoder, gt_encoder_wgt) + criterion(predicted_decoder, gt_decoder_wgt) + consistency_loss(predicted_weights, data_loader1)       # introduce lambda
                print('batch_index: {}, i: {}/{}, loss: {}'.format(batch_index,i+1,cur_batch_size,loss.item()))
                val_loss += loss.item()
                
        val_loss = val_loss/len(test_loader.sampler)
        print('Testing loss: ',val_loss)

    ### TASK 2 ###
    print('===> Target: Task-2')
    np.random.shuffle(data_index)
    data_index_ = data_index[:200]        # taking first 200

    data_sampler = SubsetRandomSampler(data_index_)
    
    dataset2 = Dataset2(y_path, y_filenames)
    data_loader2 = torch.utils.data.DataLoader(dataset2, batch_size=batch_size, sampler=data_sampler, num_workers=num_workers)

    gt_path = path + '/saved_model/autoencoding/autoencoding_model_27.pt'
       
    #Randomize indices
    np.random.shuffle(indices)
    split = int(np.floor(num_train*TEST_SIZE))
    train_index, test_index = indices[split:], indices[:split]# Making samplers for training and validation batches

    train_sampler = SubsetRandomSampler(train_index)
    test_sampler = SubsetRandomSampler(test_index)         

    # Creating data loaders
    dataset = Dataset(x1_path, x1_filenames, x2_path, x2_filenames, x3_path, x3_filenames, gt_path)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=NUM_WORKERS)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=test_sampler, num_workers=NUM_WORKERS)

    for epoch in range(1, EPOCHS+1):
        print('=> Epoch: ',epoch)
        net.train()
        for batch_index, (data, target) in enumerate(train_loader):
            (encoder_wgt_task1, encoder_wgt_task2, encoder_wgt_task3) = data
            (gt_encoder_wgt, gt_decoder_wgt) = target
            (gt_encoder_wgt, gt_decoder_wgt) = (gt_encoder_wgt.cuda(), gt_decoder_wgt.cuda())

            ################ Self-mode ################
            print('Training in Self-mode')
            net.branch1.require_grad = False
            net.branch2.require_grad = True
            net.branch3.require_grad = False

            loss_batch = 0
            cur_batch_size = encoder_wgt_task1.size()[0]

            for i in range(cur_batch_size):
                # Use append_gamma to append gama
                encoder_wgt_task1_ = append_gamma(2,encoder_wgt_task1[i])
                encoder_wgt_task1_ = encoder_wgt_task1_.cuda()
                encoder_wgt_task2_ = append_gamma(5,encoder_wgt_task2[i])
                encoder_wgt_task2_ = encoder_wgt_task2_.cuda()
                encoder_wgt_task3_ = append_gamma(3,encoder_wgt_task3[i])
                encoder_wgt_task3_ = encoder_wgt_task3_.cuda()

                # forward
                predicted_encoder, predicted_decoder = net(encoder_wgt_task1_, encoder_wgt_task2_, encoder_wgt_task3_)
                predicted_weights = torch.cat((predicted_encoder, predicted_decoder), dim=0)
                predicted_weights = predicted_weights.cuda()

                loss = criterion(predicted_encoder, gt_encoder_wgt) + criterion(predicted_decoder, gt_decoder_wgt) + consistency_loss(predicted_weights, data_loader2)       # introduce lambda
                print('batch_index: {}, i: {}/{}, loss: {}'.format(batch_index,i+1,cur_batch_size,loss.item()))
                loss_batch += loss

            # clears old gradients from the last step
            optimizer.zero_grad()
            loss_batch /= cur_batch_size
            # backward
            loss_batch.backward()
            print('Self-mode Training loss: ',loss_batch.item())         
            # fit
            optimizer.step()

            ################ Transfer-mode ################
            print('Training in Transfer-mode')
            net.branch1.require_grad = True
            net.branch2.require_grad = False
            net.branch3.require_grad = True

            loss_batch = 0
            cur_batch_size = encoder_wgt_task1.size()[0]

            for i in range(cur_batch_size):
                # Use append_gamma to append gama
                encoder_wgt_task1_ = append_gamma(2,encoder_wgt_task1[i])
                encoder_wgt_task1_ = encoder_wgt_task1_.cuda()
                encoder_wgt_task2_ = append_gamma(5,encoder_wgt_task2[i])
                encoder_wgt_task2_ = encoder_wgt_task2_.cuda()
                encoder_wgt_task3_ = append_gamma(3,encoder_wgt_task3[i])
                encoder_wgt_task3_ = encoder_wgt_task3_.cuda()

                # forward
                predicted_encoder, predicted_decoder = net(encoder_wgt_task1_, encoder_wgt_task2_, encoder_wgt_task3_)
                predicted_weights = torch.cat((predicted_encoder, predicted_decoder), dim=0)
                predicted_weights = predicted_weights.cuda()

                loss = criterion(predicted_encoder, gt_encoder_wgt) + criterion(predicted_decoder, gt_decoder_wgt) + consistency_loss(predicted_weights, data_loader2)       # introduce lambda
                print('batch_index: {}, i: {}/{}, loss: {}'.format(batch_index,i+1,cur_batch_size,loss.item()))
                loss_batch += loss

            # clears old gradients from the last step
            optimizer.zero_grad()

            loss_batch /= cur_batch_size
            loss_batch.backward() 
            print('Transfer-mode Training loss: ',loss_batch.item())        
            optimizer.step()

        # save model
        tt_model_path = path+'/tt_model/tt_model_2_'+str(epoch)+'.pt'
        torch.save(net.state_dict(), tt_model_path)

        net.eval()
        val_loss = 0
        print('Testing')
        for batch_index, (data, target) in enumerate(test_loader):
            (encoder_wgt_task1, encoder_wgt_task2, encoder_wgt_task3) = data
            (gt_encoder_wgt, gt_decoder_wgt) = target
            (gt_encoder_wgt, gt_decoder_wgt) = (gt_encoder_wgt.cuda(), gt_decoder_wgt.cuda())

            
            cur_batch_size = encoder_wgt_task1.size()[0]

            for i in range(cur_batch_size):
                # Use append_gamma to append gama
                encoder_wgt_task1_ = append_gamma(5,encoder_wgt_task1[i])
                encoder_wgt_task1_ = encoder_wgt_task1_.cuda()
                encoder_wgt_task2_ = append_gamma(2,encoder_wgt_task2[i])
                encoder_wgt_task2_ = encoder_wgt_task2_.cuda()
                encoder_wgt_task3_ = append_gamma(3,encoder_wgt_task3[i])
                encoder_wgt_task3_ = encoder_wgt_task3_.cuda()

                # forward
                predicted_encoder, predicted_decoder = net(encoder_wgt_task1_, encoder_wgt_task2_, encoder_wgt_task3_)
                predicted_weights = torch.cat((predicted_encoder, predicted_decoder), dim=0)
                predicted_weights = predicted_weights.cuda()

                loss = criterion(predicted_encoder, gt_encoder_wgt) + criterion(predicted_decoder, gt_decoder_wgt) + consistency_loss(predicted_weights, data_loader1)       # introduce lambda
                print('batch_index: {}, i: {}/{}, loss: {}'.format(batch_index,i+1,cur_batch_size,loss.item()))
                val_loss += loss.item()

        val_loss = val_loss/len(test_loader.sampler)
        print('Testing loss: ',val_loss)

    ### TASK 3 ###
    print('===> Target: Task-3')
    np.random.shuffle(data_index)
    data_index_ = data_index[:200]        # taking first 200

    data_sampler = SubsetRandomSampler(data_index_)
    
    dataset3 = Dataset3(y_path, y_filenames)
    data_loader3 = torch.utils.data.DataLoader(dataset3, batch_size=batch_size, sampler=data_sampler, num_workers=num_workers)

    gt_path = path + '/saved_model/denoising/denoising_model_27.pt'

    #Randomize indices
    np.random.shuffle(indices)
    split = int(np.floor(num_train*TEST_SIZE))
    train_index, test_index = indices[split:], indices[:split]# Making samplers for training and validation batches

    train_sampler = SubsetRandomSampler(train_index)
    test_sampler = SubsetRandomSampler(test_index)         

    # Creating data loaders
    dataset = Dataset(x1_path, x1_filenames, x2_path, x2_filenames, x3_path, x3_filenames, gt_path)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=NUM_WORKERS)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=test_sampler, num_workers=NUM_WORKERS)

    for epoch in range(1, EPOCHS+1):
        print('=> Epoch: ',epoch)
        net.train()
        for batch_index, (data, target) in enumerate(train_loader):
            (encoder_wgt_task1, encoder_wgt_task2, encoder_wgt_task3) = data
            (gt_encoder_wgt, gt_decoder_wgt) = target
            (gt_encoder_wgt, gt_decoder_wgt) = (gt_encoder_wgt.cuda(), gt_decoder_wgt.cuda())

            ################ Self-mode ################
            print('Training in Self-mode')
            net.branch1.require_grad = False
            net.branch2.require_grad = False
            net.branch3.require_grad = True

            loss_batch = 0
            cur_batch_size = encoder_wgt_task1.size()[0]

            for i in range(cur_batch_size):
                # Use append_gamma to append gama
                encoder_wgt_task1_ = append_gamma(1,encoder_wgt_task1[i])
                encoder_wgt_task1_ = encoder_wgt_task1_.cuda()
                encoder_wgt_task2_ = append_gamma(2,encoder_wgt_task2[i])
                encoder_wgt_task2_ = encoder_wgt_task2_.cuda()
                encoder_wgt_task3_ = append_gamma(5,encoder_wgt_task3[i])
                encoder_wgt_task3_ = encoder_wgt_task3_.cuda()

                # forward
                predicted_encoder, predicted_decoder = net(encoder_wgt_task1_, encoder_wgt_task2_, encoder_wgt_task3_)
                predicted_weights = torch.cat((predicted_encoder, predicted_decoder), dim=0)
                predicted_weights = predicted_weights.cuda()

                loss = criterion(predicted_encoder, gt_encoder_wgt) + criterion(predicted_decoder, gt_decoder_wgt) + consistency_loss(predicted_weights, data_loader3)       # introduce lambda
                print('batch_index: {}, i: {}/{}, loss: {}'.format(batch_index,i+1,cur_batch_size,loss.item()))
                loss_batch += loss

            # clears old gradients from the last step
            optimizer.zero_grad()

            loss_batch /= cur_batch_size
            loss_batch.backward()   
            print('Self-mode Training loss: ',loss_batch.item())      
            optimizer.step()

            ################ Transfer-mode ################
            print('Training in Transfer-mode')
            net.branch1.require_grad = True
            net.branch2.require_grad = True
            net.branch3.require_grad = False

            loss_batch = 0
            cur_batch_size = encoder_wgt_task1.size()[0]

            for i in range(cur_batch_size):
                # Use append_gamma to append gama
                encoder_wgt_task1_ = append_gamma(1,encoder_wgt_task1[i])
                encoder_wgt_task1_ = encoder_wgt_task1_.cuda()
                encoder_wgt_task2_ = append_gamma(2,encoder_wgt_task2[i])
                encoder_wgt_task2_ = encoder_wgt_task2_.cuda()
                encoder_wgt_task3_ = append_gamma(5,encoder_wgt_task3[i])
                encoder_wgt_task3_ = encoder_wgt_task3_.cuda()

                # forward
                predicted_encoder, predicted_decoder = net(encoder_wgt_task1_, encoder_wgt_task2_, encoder_wgt_task3_)
                predicted_weights = torch.cat((predicted_encoder, predicted_decoder), dim=0)
                predicted_weights = predicted_weights.cuda()

                loss = criterion(predicted_encoder, gt_encoder_wgt) + criterion(predicted_decoder, gt_decoder_wgt) + consistency_loss(predicted_weights, data_loader3)       # introduce lambda
                print('batch_index: {}, i: {}/{}, loss: {}'.format(batch_index,i+1,cur_batch_size,loss.item()))
                loss_batch += loss

            # clears old gradients from the last step
            optimizer.zero_grad()

            loss_batch /= cur_batch_size
            loss_batch.backward() 
            print('Transfer-mode Training loss: ',loss_batch.item())        
            optimizer.step()

        # save model
        tt_model_path = path+'/tt_model/tt_model_3_'+str(epoch)+'.pt'
        torch.save(net.state_dict(), tt_model_path)

        net.eval()
        val_loss = 0
        print('Testing')
        for batch_index, (data, target) in enumerate(test_loader):
            (encoder_wgt_task1, encoder_wgt_task2, encoder_wgt_task3) = data
            (gt_encoder_wgt, gt_decoder_wgt) = target
            (gt_encoder_wgt, gt_decoder_wgt) = (gt_encoder_wgt.cuda(), gt_decoder_wgt.cuda())

            
            cur_batch_size = encoder_wgt_task1.size()[0]

            for i in range(cur_batch_size):
                # Use append_gamma to append gama
                encoder_wgt_task1_ = append_gamma(5,encoder_wgt_task1[i])
                encoder_wgt_task1_ = encoder_wgt_task1_.cuda()
                encoder_wgt_task2_ = append_gamma(2,encoder_wgt_task2[i])
                encoder_wgt_task2_ = encoder_wgt_task2_.cuda()
                encoder_wgt_task3_ = append_gamma(3,encoder_wgt_task3[i])
                encoder_wgt_task3_ = encoder_wgt_task3_.cuda()

                # forward
                predicted_encoder, predicted_decoder = net(encoder_wgt_task1_, encoder_wgt_task2_, encoder_wgt_task3_)
                predicted_weights = torch.cat((predicted_encoder, predicted_decoder), dim=0)
                predicted_weights = predicted_weights.cuda()

                loss = criterion(predicted_encoder, gt_encoder_wgt) + criterion(predicted_decoder, gt_decoder_wgt) + consistency_loss(predicted_weights, data_loader1)       # introduce lambda
                print('batch_index: {}, i: {}/{}, loss: {}'.format(batch_index,i+1,cur_batch_size,loss.item()))
                val_loss += loss.item()

        val_loss = val_loss/len(test_loader.sampler)
        print('Testing loss: ',val_loss)

train()

