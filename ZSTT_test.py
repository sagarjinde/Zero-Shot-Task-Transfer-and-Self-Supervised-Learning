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
from PIL import Image
# %matplotlib inline

num_workers = 0
batch_size = 32
test_size = 0.02
num_models = 48

NUM_TASKS = 3
ENCODER_DIMENSION = 508
DECODER_DIMENSION = 503
INPUT_DIM = ENCODER_DIMENSION + 1
HIDDEN_DIM = 256*3
OUTPUT_DIM = 256
TT_HIDDEN_DIM = 1024
TT_OUTPUT_DIM = 512

class Dataset4(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, x_path, y_path, x_filenames, y_filenames):
        'Initialization'
        self.x_path = x_path
        self.y_path = y_path
        self.x_filenames = x_filenames
        self.y_filenames = y_filenames

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.x_filenames)

  def __getitem__(self, index):
        'Generates one sample of data'
        X = Image.open(os.path.join(self.x_path, self.x_filenames[index]))
        x = transforms.ToTensor()(X)
        Y = Image.open(os.path.join(self.y_path, self.y_filenames[index]))
        y = transforms.ToTensor()(Y)

        return x, y

class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, x1_path, x1_filenames, x2_path, x2_filenames, x3_path, x3_filenames):
        'Initialization'
        self.x1_path = x1_path
        self.x1_filenames = x1_filenames
        self.x2_path = x2_path
        self.x2_filenames = x2_filenames
        self.x3_path = x3_path
        self.x3_filenames = x3_filenames

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

        return [encoder_wgt_task1, encoder_wgt_task2, encoder_wgt_task3]

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

##########################################################################
# weights as input to TTNet
path = os.path.dirname(os.path.realpath(__file__))

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

# Creating data loaders
sampler = SubsetRandomSampler(indices)
dataset = Dataset(x1_path, x1_filenames, x2_path, x2_filenames, x3_path, x3_filenames)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler, pin_memory=False, num_workers=0)

##########################################################################
branch1 = Branch(); branch1.cuda()
branch2 = Branch(); branch2.cuda()
branch3 = Branch(); branch3.cuda()

tt_net = TaskTransferNet(branch1, branch2, branch3)
tt_net.cuda()

# load TTNet weights
tt_net.load_state_dict(torch.load(path+'/tt_model/tt_model_3_25.pt'))

##########################################################################

parent_path = path+'/taskonomy-sample-model-1'
x_dir = 'rgb'
y_dir = 'normal'

x_path = os.path.join(parent_path, x_dir)
x_filenames = os.listdir(x_path)
x_filenames.sort(key=lambda f: int(re.sub('\D', '', f)))

y_path = os.path.join(parent_path, y_dir)
y_filenames = os.listdir(y_path)
y_filenames.sort(key=lambda f: int(re.sub('\D', '', f)))

model_num = len(x_filenames)
model_indices = list(range(model_num))
np.random.shuffle(model_indices)
model_indices = model_indices[:100]        # taking 100 pics

model_sampler = SubsetRandomSampler(model_indices)# Creating data loaders
model_dataset = Dataset4(x_path, y_path, x_filenames, y_filenames)
model_loader = torch.utils.data.DataLoader(model_dataset, batch_size=batch_size, sampler=model_sampler, num_workers=num_workers)

##########################################################################
# loss function (cross entropy loss)
criterion = nn.MSELoss()

model = Net()
print(model)
model.cuda()

##########################################################################
tt_net.eval()
for batch_index, data in enumerate(loader):
    (encoder_wgt_task1, encoder_wgt_task2, encoder_wgt_task3) = data
    encoder_wgt_task1_ = append_gamma(2,encoder_wgt_task1[0])
    encoder_wgt_task1_ = encoder_wgt_task1_.cuda()
    encoder_wgt_task2_ = append_gamma(1,encoder_wgt_task2[0])
    encoder_wgt_task2_ = encoder_wgt_task2_.cuda()
    encoder_wgt_task3_ = append_gamma(1,encoder_wgt_task3[0])
    encoder_wgt_task3_ = encoder_wgt_task3_.cuda()

    # forward
    predicted_encoder, predicted_decoder = tt_net(encoder_wgt_task1_, encoder_wgt_task2_, encoder_wgt_task3_)
    model_predicted_weights = torch.cat((predicted_encoder, predicted_decoder), dim=0)
    model_predicted_weights = model_predicted_weights.cuda()

    nn.utils.vector_to_parameters(model_predicted_weights,model.parameters())

    model_loss = 0
    model.eval()        
    for model_batch_index, (model_data, model_target) in enumerate(model_loader):
        # moves tensors to GPU
        model_data, model_target = model_data.cuda(), model_target.cuda()
        # forward pass
        model_output = model(model_data)
        # loss in batch
        loss = criterion(model_output, model_target)
        # update validation loss
        model_loss += loss.item()*model_data.size(0)
            
    # average loss calculations
    model_loss = model_loss/len(model_loader.sampler)
        
    # Display loss statistics
    print(f'Loss: {round(model_loss, 6)}')


