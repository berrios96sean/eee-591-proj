# eee-591-proj

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

## Activate env 

create and activate a conda environment with the following 
```
conda create --name yolov5_env python=3.11 -y
conda activate yolov5_env

```

## install dependencies 
go to or cd to yolov5 directory and run the following 

```
conda install pip -y
pip install -r requirements.txt
```

this should work without issues. 
