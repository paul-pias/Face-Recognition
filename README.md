# Face Recognition using ARCFACE-Pytorch

## Introduction
 This repo contains face_verify.py and app.py which is able to perform the following task -
 - Detect faces from an image, video or in webcam and perform face recogntion.
 - app.py was used to deploy the project.

## User Instruction
After downloading the project first you have to install the following libraries.
### Installation
You can install all the dependencies at once by running the following command from your terminal.
``` python
    $ pip install -r requirements.txt
```
##### For the installation of torch using "pip" run the following command

``` python
    $ pip3 install torch===1.2.0 torchvision===0.4.0 -f https://download.pytorch.org/whl/torch_stable.html
```
### Project Setup
You have to add the following files to the "data/faces_emore" folder. 

    faces_emore/
                ---> agedb_30
                ---> calfw
                ---> cfp_ff
                --->  cfp_fp
                ---> cfp_fp
                ---> cplfw
                --->imgs
                ---> lfw
                ---> vgg2_fp

To get these files, first you need to download the [MS1M](https://arxiv.org/abs/1607.08221) dataset either from
- [emore dataset @ Dropbox](https://www.dropbox.com/s/wpx6tqjf0y5mf6r/faces_ms1m-refine-v2_112x112.zip?dl=0)
- [emore dataset @ BaiduDrive](https://pan.baidu.com/s/1eXohwNBHbbKXh5KHyItVhQ)

After unzipping the downloaded file execute the following command. It will take few hours depending on your system configuration.

```python
    $ python prepare_data.py
```
However, if you already have those files you can excute 

```python
    $ python app.py
 ```
 and go to the following url from your web browser.
 ```url
http://localhost:5000
```
<hr>
<hr>

Now if you want to train with your custom dataset, you need to follow the following steps.

#### Dataset preparation 
First organize your images within the following manner-
    
    data/
        raw/
             name1/
                 photo1.jpg
                 photo2.jpg
                 ...
             name2/
                 photo1.jpg
                 photo2.jpg
                 ...
             .....
now run the following command

```python
$ python .\create-dataset\align_dataset_mtcnn.py data/raw/ data/processed --image_size 112
```

You will see a new folder inside the data directory named <b> "processed" </b> which will hold all the images that contains only faces of evey user. If more than 1 image appears in any folder for a person, average embedding will be calculated. 

Copy all the folders of the users under the <b>data/processed</b> folder and paste in the <b>data/facebank</b> folder.

