# Segmentation of Optic Disc and Cup Based on SegNet and Domain Adaptation
## File Structure
./  
|----dataset-----------------------------------------------------------------contains datasets and path files  
|　　　|-----(directory)-------------------------------------------dataset directory (need to be downloaded)  
|　　　|-----da-----------------------------------------------------------domain adaptation images path file  
|　　　|-----Disc_Crop.py---------------------------------------------------code for cropping images and labels  
|　　　|-----Image3to255.py-------------------------------------------------change labels from 0, 1, 2 to 255, 128, 0  
|　　　|-----Image255to3.py-------------------------------------------------change labels from 255, 128, 0 to 0, 1, 2   
|----evaluation-----------------------------------------------------------------contains evaluation codes     
|　　　|-----evaluate_single_submission.py-------------------------------------------------main code evaluation  
|　　　|-----evaluation_metrics_for_classification.py-----------------------metrics for classification evaluation  
|　　　|-----evaluation_metrics_for_fovea_location.py-----------------------metrics for fovea_location evaluation   
|　　　|-----evaluation_metrics_for_segmentation.py-------------------------metrics for segmentation evaluation   
|　　　|-----file_management.py------------------------------------------------------------------read files  
|　　　|-----leaderboard_criteria.py--------------------------------------------------not used in this project   
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
* We referred to the [tf-SegNet](https://github.com/tkuanlun350/Tensorflow-SegNet "Title") implemented by [Tseng Kuan Lun](https://github.com/tkuanlun350 "title"). We replaced the devonc layer with unpool layer used in the original [Caffe version SegNet](https://github.com/alexgkendall/caffe-segnet "Title");
* We replaced the weighted loss with Jaccard loss implemented by ourselves. We get the inspiration from Hongyi Guo and Yuqiao He, who also took this project, when they deliver their mid-term presentation;
* We cropped the origianl data provided by our T.A. to do data augmentation;
* We implemented [ADDA](http://openaccess.thecvf.com/content_cvpr_2017/papers/Tzeng_Adversarial_Discriminative_Domain_CVPR_2017_paper.pdf "Title") to achieve domain adaptation, so that our model is better on other datasets such as Validation400;(Other works that didn't result in performance improvement is not listed here)
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
`python main.py --mode datest --ckpt adda/tar_model-99999`

## Dataset
The original dataset is provided by our T.A., which contains opthalmoscopy photographs (2124x2056).  
- [Original Dataset](https://pan.baidu.com/share/init?surl=AIhsyDsmYeg84izrMR0eNQ "Title") Validation password: m53z  

This is our processed dataset. We used the `mogirfy` command to change the size to 480x480. Also, this dataset includes crop images for data augmentation. 
- [Processed Dataset](https://pan.baidu.com/s/15B40Q4Qz5se3yV12UiLJbw "Title") Validation password: avkd  
Usage: unzip and move all files into `./dataset/`.  

## Pre-trained models
- [Segnet trained model](https://pan.baidu.com/s/16WHkvr4wdll6sT3Sc_A7_g "Title") Validation password: cqeb  
- [ADDA trained model](https://pan.baidu.com/s/1bBLX5BMn0q_0Qibq9-vGpA "Title") Validation password: ge7w  
