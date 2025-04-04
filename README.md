# eee-591-proj
Introduction:

The goal of this project is to take a large dataset of ~8000 thermal images from a FLIR Tau2 camera, and train the "You Only Look Once" (YOLO) Object detection algorithm, implemented with PyTorch, to be able to detect and classify objects such as a car or person in new thermal video or images. 

The train function of YOLO takes a YAML file that directs it to folders containing training and validation images and labels.

The detect function will use the weights provided by train function to classify and label new images/videos.

To improve training speed we will use the ASU research computer labs SOL Supercomputer.

Request access to SOL here: https://rto.asu.edu/forms/request-access-to-the-asu-supercomputers/

New User Guide for SOL here: https://asurc.atlassian.net/wiki/spaces/RC/pages/1905721457/New+User+Guide


The original dataset of images and annotations is found on kaggle. Link Here:

We added the dataset to scratch so it can be accessed by the SOL Computer

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

# Setting up a mini project

## Cleaning generated labels and mini project
I've updated this step to grab your userid and clean both labels and mini projects you've made using just this command 

```
make clean
```

>> Important NOTE:
>> I have update this process to use a single Make command which I'll explain below in detail. These scripts have been modifed to use only training data instead of validation data or images this has significantly improved results.

The files mentioned below can be run with a single command from the src directory

```
make run-all
```
The default img_num for this is 100. You can change this with the make command below 

```
make run-all IMG_NUM=1000
```

another flag has been added for the scripts and the make command as shown below

```
make run-all IMG_NUM=1000 START_IDX=20
```

This is an important step since the data set has over 14000 images training over the entire dataset is not really ideal. The start index flag allows us to modify the start location of the annotations and validations images we use for training for example setting `START_IDX=20` will start the process at FLIR_00020.jpeg instead of FLIR_00001.jpeg. I have went through a few runs and verified this works as intended and the results with just 30 epochs are pretty good so for with a `IMG_NUM=2000` it's also importatn to note that if we specify 2000 images I've updated the script to do a 60/40 split of the training data to the /train and /val folders instead of usingn images only from the /val folder. This has given better results so far. The idea behind the start index is so we can do cross validation to see if we have any success with that approach for improving precision. I have added a folder in the google drive with something like BEST-RESULTS and you can see the progress using 2000 images starting at FLIR_00001. 



# Running the model 

>> NOTE: Before running the model go to the root directory and run the following Command after you have done the setup for the mini project.
>> ```
>> make setup-run
>> ```

running the model is straightforward however some considerations. Yolov5 will run with CUDA automatically if it detects it, a good way to ensure you have CUDA avaiable is to runn the following command If it prints TRUE go ahead and run the model, otherwise it will run on CPU and be MUCH slower. If you are using Sol CUDA should be available if you have requested a GPU when you created the session. 

```
python -c "import torch; print(torch.cuda.is_available())"
```

Before running the run_yolo.py script make sure you update the parameters you want to use inside the file before the run then run 

```
python run_yolo.py
```

# Additional Information for later use 
IN the source directory I have a classes.txt file. This file is used for the annotation conversion.. the contents look something like this 

```
4
person
car
bicycle
dog
```

The reason behind this is because when we create the annotations for the images we only want to include the images for the classifications we are using. If you want to add more classes you can add them in this text file just keep in mind that the first line indicates the number of classes we are using fof example the `4` above indicates we have 4 classes listed in the lines below person, car, bicycle and dog. 

Another important consideration is the order of the classes in the `classes.txt` file should match what is in the yaml. Rather than update the yaml file update the `gen_yaml.py` where the classes are listed for example this is directly form the script below 

```
nc: 4  # Number of classes
names: ['person', 'car', 'bicycle', 'dog']  # Class names (corrected 'dogs' → 'dog' for consistency)
```

change `nc: ` to the number of classes you are using and the `names: ` to match the order of the classes in your `classes.txt` file. Then you can just run 

```
python gen_yaml.py
```

or 

```
make setup-run
```

as the command for generating the yaml is in the make command also. 
