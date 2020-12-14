# Zero-Shot Task Transfer and Self-Supervised Learning

## Objective
Computed model parameters of a supervised task without any explicit supervision

## Get Started
This work is a modified version of [Zero-Shot Task Transfer](https://openaccess.thecvf.com/content_CVPR_2019/papers/Pal_Zero-Shot_Task_Transfer_CVPR_2019_paper.pdf).

Input tasks are self-supervised tasks and output tasks are supervised task

Go through the [report](https://github.com/sagarjinde/Zero-Shot-Task-Transfer-and-Self-Supervised-Learning/blob/master/report.pdf) for detailed explanation of the model

### Implementation Details
- Input tasks considered are:
	- Jigsaw
	- Autoencoding
	- Denoising
- Output task is:
	- Surface Normal

Correlation matrix considered is the same as [Zero-Shot Task Transfer](https://openaccess.thecvf.com/content_CVPR_2019/papers/Pal_Zero-Shot_Task_Transfer_CVPR_2019_paper.pdf).

![correlation matrix](https://github.com/sagarjinde/Zero-Shot-Task-Transfer-and-Self-Supervised-Learning/blob/master/figs/correlation_matrix.png)

#### Training the model

![train mode](https://github.com/sagarjinde/Zero-Shot-Task-Transfer-and-Self-Supervised-Learning/blob/master/figs/train_model.png)

Model is trained in two modes:
- Self Mode
- Transfer mode

#### Self Mode
Backprop only W<sub>i</sub> and W<sub>common</sub>

#### Transfer Mode
Backprop only W<sub>!i</sub> and W<sub>common</sub>

#### Testing the model
During testing, target task is an unknown supervised task

## Results

| Loss using weights generated by TTNet | Loss using weights learnt by supervised learning |
| --- | --- |
| 0.125560 | 0.180921 |
| 0.123371 | 0.180921 | 
| 0.122083 | 0.180921 |

**Model generated by TTNet outperforms model learnt by supervised learning by 32%**

**Note:** TTNet means Task Transfer Network

## Running the code

`Python version: 3.6.8`

### Create a virtual environment (Recommended but optional)
Install virtualenv  : `sudo apt-get install virtualenv` </br>
Create virtualenv   : `virtualenv --system-site-packages -p python3 ./venv` </br>
Activate virtualenv : `source venv/bin/activate` </br>

### Install Requirements
Run `pip install -r requirements.txt`

**Note:** If you get `ModuleNotFoundError: No module named 'matplotlib._path'` error, run `pip3 install --upgrade matplotlib`

### Download Image Dataset
Download `taskonomy-sample-model-1` from [Google Drive link](https://drive.google.com/drive/folders/1vRxpGVJxySb4myYSKuoMOHQMxj8ZItVv?usp=sharing) and save it inside main folder

### How to train model
Run `python ZSTT_train.py` </br>
This will store output (generated weights of target model) in a folder called `tt_model`

### How to test model
Run `python ZSTT_test.py` </br>
This will use weight of target model generated by TTNet which is stored in `tt_model` to generate images

## Custom Dataset

- Make changes in the architecture of known and unknown tasks
- Run all the task modified python files. ex: `python jigsaw.py`
- This will create `.pt` files inside `input_model` or `saved_model` depending on weather the task was target task or not. `.pt` files store weights of model

**Note:** Model architecture for all tasks considered should be the same 

Feel free to add your custom task, but make sure to modify `ZSTT_train` and `ZSTT_test` where ever required.

**Note:** The present version of the code is only using `jigsaw`, `autoencoding` and `denoising` as input task. Initial plan was to use `colorization`, `edge2d` and 
`keypoint2d` as well, but we realized it later on that the architecture of all tasks should remain same. Hence we did not use them.

## Future work
Using models of different architectures in TTNet