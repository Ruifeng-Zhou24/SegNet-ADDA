# Segmentation of Optic Disc and Cup Based on SegNet and Domain Adaptation
## File Structure
./  
|----dataset-----------------------------------------------------------------contains datasets and path files  
|　　　|-----(directory)-------------------------------------------dataset directory (need to be downloaded)  
|　　　|-----da.txt-------------------------------------------------------domain adaptation images path file  
|　　　|-----test.txt---------------------------------------------------------------------test images path file  
|　　　|-----train.txt--------------------------------------------------------------------train images path file  
|　　　|-----val.txt----------------------------------------------------------------validation images path file  
|----logs-----------------------------contains training and domain adaptation logs (automatically generated)  
|　　　|----adda--------------------------------------------------------contains domain adaptation log files  
|　　　|　　　|----(files)---------------------------------------------------------domain adaptation log files  
|　　　|----(files)----------------------------------------------------------------training adaptation log files  
|----model-------------------------------------------------------------------------contains network models  
|　　　|----models.py-------------------------------------------------------SegNet model and ADDA model  
|----outputs--------------------------------------------contains prediction images (automatically generated)  
|　　　|----adda-----------------------------------------------------------contains ADDA prediction images  
|　　　|　　　|----(files)------------------------------------------------------------ADDA prediction images  
|　　　|----(files)------------------------------------------------------------------SegNet prediction images  
|----util--------------------------------------------------------------------utility functions and dataset class  
|　　　|----input.py----------------------------------------------------------------------------dataset class  
|　　　|----utils.py---------------------------------------------------------------------------utility functions  
|----main.py-------------------------------------------------------program entrance and arguments parsing

## Modified tf-SegNet Structure
![alt text](/SegNet_modified.png "Title")

## Our Contributions
* We referred to the tf-SegNet implemented by xxx (wang zhi). We replaced the devonc layer with unpool layer used in the original Caffe version SegNet;
* We replaced the weighted loss with Jaccard loss implemented by ourselves. We get the inspiration from Hongyi Guo and Yuqiao He, who also took this project, when they deliver their mid-term presentation;
* We cropped the origianl data provided by our T.A. to do data augmentation;
* We implemented ADDA to achieve domain adaptation, so that our model is better on other datasets such as Validation400;(Other works that didn't result in performance improvement is not listed here)
* We completed a survey regarding segmentation of optic disc and cup in collaboration with Hongyi Guo and Yuqiao He's group.

## Requirements
* python 2.7
* tensorflow-gpu 1.8
* numpy
* pillow
* scikit-image

## Setting up the Environment
We recommend you to run our project in Anaconda3. Use the following commands to set up your env.
- `conda create -n segnet python=2.7 tensorflow-gpu=1.8`
- `source activate segnet`
- `conda install pillow`
- `conda install scikit-image`


## Usage
- train segnet  
`python main.py --mode train --max_steps 20000`

- test segnet  
`python main.py --mode test --ckpt model-19999`

- finetune segnet  
`python main.py --mode finetune --ckpt model-19999 --max_steps 10000 --learning_rate 0.0001`

- train adda  
`python main.py --mode da --ckpt model-19999 --max_steps 100000`

- test adda  
`python main.py --mode datest --ckpt tar_model-99999`
