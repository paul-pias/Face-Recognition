# Face Recognition using ARCFACE-Pytorch

## Introduction
 This repo contains face_verify.py and app.py which is able to perform the following task -
 - Detect faces from an image, video or in webcam and perform face recogntion.
 - app.py was used to deploy the project.
 
## Required Files
- requirements.txt
- pretrained model [IR-SE50 @ Onedrive](https://onedrive.live.com/?authkey=%21AOw5TZL8cWlj10I&cid=CEC0E1F8F0542A13&id=CEC0E1F8F0542A13%21835&parId=root&action=locate) or [Mobilefacenet @ OneDrive](https://onedrive.live.com/?authkey=%21AIweh1IfiuF9vm4&cid=CEC0E1F8F0542A13&id=CEC0E1F8F0542A13%21836&parId=root&o=OneUp).
- Custom dataset
- Newly Trained model (facebank.pth and names.npy)


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

#### pre-trained model
Although i provided the pretrained model in the <b> work_space/model </b> and <b> work_space/save </b> folder, if you want to download the models you can follow the following url:

- [IR-SE50 @ BaiduNetdisk](https://pan.baidu.com/s/12BUjjwy1uUTEF9HCx5qvoQ)
- [IR-SE50 @ Onedrive](https://onedrive.live.com/?authkey=%21AOw5TZL8cWlj10I&cid=CEC0E1F8F0542A13&id=CEC0E1F8F0542A13%21835&parId=root&action=locate)
- [Mobilefacenet @ BaiduNetDisk](https://pan.baidu.com/s/1hqNNkcAjQOSxUjofboN6qg)
- [Mobilefacenet @ OneDrive](https://onedrive.live.com/?authkey=%21AIweh1IfiuF9vm4&cid=CEC0E1F8F0542A13&id=CEC0E1F8F0542A13%21836&parId=root&o=OneUp)

I have used the <b>IR-SE50</b> as the pretrained model to train with my custom dataset. You need to copy the pretrained model and save it under the <b> work_space/save </b> folder as <b> model_final.pth</b>

#### Newly trained model
In the <b> data/facebank </b> you will find a trained model named <b> "facebank.pth" </b> which contains the related weights and "names.npy" contains the corresponding labels of the users that are avialable in the facebank folder. For instance in this case
the <b> facebank </b> folder will look like this :-

    facebank/
                ---> Chandler
                ---> Joey
                ---> Monica
                ---> Phoebe
                ---> Pias
                ---> Rachel
                ---> Raihan
                ---> Ross
                ---> Samiur
                ---> Shakil
                ---> facebank.pth
                ---> names.npy

If you have the "facebank.pth" and "names.npy" files in the <b>data/facebank</b> you can execute the following command to see the demo.

```python
    $ python app.py
 ```
 and go to the following url from your web browser.
 ```url
http://localhost:5000
```


<hr>
Note: If you want to run the inference on a video, download a video of related persons (Person that you trained the model with) and replace 0 in the line number 43 of <b> face_verify.py </b> with the path of your video file. For this code you can run the inference on any video of <b> Friends</b> tv series.
<hr>

#### Now if you want to train with your custom dataset, you need to follow the following steps.

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

You will see a new folder inside the data directory named <b> "processed" </b> which will hold all the images that contains only faces of each user. If more than 1 image appears in any folder for a person, average embedding will be calculated. 

After executing the script new images for each user in the processed folder will look something like this.
<p align="center"> 
<b> Cropped Images of faces </b>
    <img src ="http://muizzer07.pythonanywhere.com/media/files/Picture1.png">
</p> 

Copy all the folders of the users under the <b>data/processed</b> folder and paste in the <b>data/facebank</b> folder.


Now to train with your dataset, you need to set <b> args.update == True </b> in line 35 of face_verify.py . After training you will get a new facebank.pth and names.npy in your data/facebank folder which will now only holds the weights and labels of your newly trained dataset. Once the training is done you need to reset <b> args.update==False</b>.
However, if this doesn't work change the code in following manner-
#### Old Code 
```python
    if args.update:
        targets, names = prepare_facebank(conf, learner.model, mtcnn, tta = args.tta)
        print('facebank updated')
    else:
        targets, names = load_facebank(conf)
        print('facebank loaded')
```
#### New Code 
Only keep the follwing lines for training, once the training is done just replace it with the old code.
```python
        targets, names = prepare_facebank(conf, learner.model, mtcnn, tta = args.tta)
        print('facebank updated')
````
Or you can simply pass a command line arguement such as below if there is new data to train.
```python
   $python face_verify.py -u
```
Here the -u parse the command to update the facebank.pth and names.npy.

Now you are ready to test the systen with your newly trained users by running-

```python
    $ python app.py
```

### Note: You can train with custom dataset as many time as you want, you will only require any of the pre-trained model to train with your custom dataset to generate the <b>facebank.pth</b> and <b>names.npy</b>. Once you get this two files you are ready to test the face recogniton.


<hr>

### Retrain the pre-trained model

 If you want to build a new pre-trained model like [IR-SE50 @ Onedrive](https://onedrive.live.com/?authkey=%21AOw5TZL8cWlj10I&cid=CEC0E1F8F0542A13&id=CEC0E1F8F0542A13%21835&parId=root&action=locate) and reproduce the result, you will need the large files which contains several dataset of faces under the <b> data/faces_emore </b>.
 
 To get these files, first you need to download the [MS1M](https://arxiv.org/abs/1607.08221) dataset from any of the following url-
- [emore dataset @ Dropbox](https://www.dropbox.com/s/wpx6tqjf0y5mf6r/faces_ms1m-refine-v2_112x112.zip?dl=0)
- [emore dataset @ BaiduDrive](https://pan.baidu.com/s/1eXohwNBHbbKXh5KHyItVhQ)

After unzipping the downloaded file execute the following command. It will take few hours depending on your system configuration.

```python
    $ python prepare_data.py
```
After that you will see the following files to the "data/faces_emore" folder. 

    faces_emore/
                ---> agedb_30
                ---> calfw
                ---> cfp_ff
                ---> cfp_fp
                ---> cfp_fp
                ---> cplfw
                ---> imgs
                ---> lfw
                ---> vgg2_fp

To know the further training procedure you can see the details in this [InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch) repository.

## References
- [Arcface](https://arxiv.org/pdf/1801.07698.pdf)
- [InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch)
- [The one with Face Recognition.](https://towardsdatascience.com/s01e01-3eb397d458d)
