# eee-591-proj

The goal of this project is to take a large dataset of ~8000 thermal images from a FLIR Tau2 camera, and train the "You Only Look Once" (YOLO) Object detection algorithm, implemented with PyTorch, to be able to detect and classify objects such as a car or person in new thermal video or images. 

To improve training speed we will use the ASU research computer labs SOL Supercomputer. 
New User Guide for SOL here: https://asurc.atlassian.net/wiki/spaces/RC/pages/1905721457/New+User+Guide

Instructions for training on Local Device (for those who are not ASU students) is also provided.

Step 1.01 : login to SOL
Step 1.02 : 


# Important Note 

Don't ever push any changes made inside of the Yolov5 submodule. Just run the makefile command in the root directory to move the yaml file to the data folder of yolov5 before moving forward with the following command 

```
make add-yaml
```

Verify this is in the yolov5/data dir then proceed. 

Do not ever to a `git add yolov5` this will screw things up. 

# Dataset Directory on Sol Server
Check that you have access to the scratch directory for the dataset run this and make sure you aren't seeing permission errors. 

```
cd /scratch/sfberrio/FLIR_ADAS_1_3/
```

# Setting Up YOLOv5 with Miniconda and `requirements.txt` on Linux

Sol Server seems to be using python 3.6 for me.. to install dependencies for yolov5 newer versions above 3.7 are needed the following is what worked for me. 

---

## **1. Install Miniconda**
If Miniconda is not installed, download and install it with the following commands:

```bash
# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Run the installer
bash Miniconda3-latest-Linux-x86_64.sh
```
Follow the installation prompts leave the directory location empty to use the defualt install location. then once finished do the following 

```
source ~/.bashrc
```
check that conda installed by doing `conda --version`

## 2. Activate env 

create and activate a conda environment with the following 
```
conda create --name yolov5_env python=3.11 -y
conda activate yolov5_env

```

## 3. install dependencies 
go to or cd to yolov5 directory and run the following 

```
conda install pip -y
pip install -r requirements.txt
```

this should work without issues. 

# Setting up a mini project

I've created some scripts to automate this process in the src directory and updated the original json conversion script. These only work if you have access to the /scratch directroy mentioned. Otherwise you will just need to adjust the location of the Dataset for where you have it stored. 

## convert_json_to_yolo.py

This has been updated to take in a parameter `--img_num 20` for example will create a folder `yolo_labels` in the scratch directory for both train and val. It will create the number of labels specified in `--img_num` otherwise if you don't specify a number it will generate all labels for every image example use below. 
```
python convert_json_to_yolo.py --img_num 20
```

I've also included a Makefile to delete those folders if you want to create more lables at some point, in the src directory just run 
```
make delete-labels
```

## create_thermal_8_bit_mini.py
This script creates a folder named thermal_8_bit_mini in both train and val folders in the dataset location for the scratch directory. If you don't have access to the scratch directory then just modify it to where your dataset is. The same `--img_num` flag is used however you must specify a number for this script to work. Example use below
```
python create_thermal_8_bit_mini.py --img_num 20
```

A makefile command also exist to delete this directories if you want to test different numbers of images just run the following inside the src directory 
```
make delete-mini
```
After you have done this the setup is complete just make sure if you aren't using the dataset location you update the YAML file in `yolov5/data`

# Running the model 

running the model is straightforward however some considerations. Yolov5 will run with CUDA automatically if it detects it, a good way to ensure you have CUDA avaiable is to runn the following command If it prints TRUE go ahead and run the model, otherwise it will run on CPU and be MUCH slower. If you are using Sol CUDA should be available if you have requested a GPU when you created the session. 

```
python -c "import torch; print(torch.cuda.is_available())"
```

it should be noted that I ran into alot of issues when using the oringinal conversion script.. before running make sure the .txt files have the same filename as the .jpeg files.. I have updated the script to do this as well but just in case you follow a different process than me 

```
python train.py --img 640 --batch 4 --epochs 10 --data /path/to/yolov5/data/thermal_image_dataset.yaml --weights yolov5s.pt --cache
```

>> Note: Make sure when you run this you are still in the yolov5_env if you are using the conda environment. Otherwise you will keep running into dependecy issues. 
