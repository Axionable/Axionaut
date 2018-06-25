## AxionautV2 - DIY Robocar Tricks and Hacks
The Axionaut is the winner of the second [IronCar France](http://ironcar.org) robocar competition. The following repository contains all the tips and tricks we used to win. The final Keras model achieves more than 98% of accuracy on the validation set. The provided autopilot model will work seamlessly with Vincent's [IronCar](https://github.com/vinzeebreak/ironcar) framework.


## Introduction
The Axionaut is based on a 4x4 Radio Controlled (RC) monster truck chassis, the complete bill of materials is available here:
https://www.axionable.com/axionaut-termine-1er-de-la-deuxieme-course-iron-car/

The following code and documentation will guide you through the best practices and tricks to train the car. All strategies were defined both searching on the Internet and by experimentation. So, are you ready to win your next race?


## Finding the right camera position
Make sure to put the camera such that the three lanes are always visible. Also, crop the original image to keep only the track. The position and crop level depends on your vehicle and the camera you are using. In our case, the image was cropped to [90, 250, 3] and the camera was placed at 20cm from the floor. We recommend the use of [eyefish cameras.](https://www.amazon.fr/Waveshare-Raspberry-Camera-Fisheye-Raspberry-pi/dp/B00RMV53Z2/ref=pd_cp_23_3?_encoding=UTF8&psc=1&refRID=7JBTZQTNFRVC34PY6J4X)

It is normally easy to know when the camera is not in the right position. Here an useful example, make sure to check your training data before training your model. Once you found a good fit, fix the camera.
![alt text](https://github.com/Axionable/AxionautV2/blob/master/Docs/camera_adj.png)


## Training the vehicle
Once you found the right camera position, it is necessary to build your own dataset. We built ours with approximately 60K images with labels verified by inspection. Taking the data directly from the car can add noise due contradictory examples and lack of synchronization, as we make mistakes or anticipate curves while driving. To avoid that, we took images directly from the car and we assigned the labels after using a script. Finally, all labels were manually inspected to assure its quality. The final dataset (2.9GB) including both IronCar track and proper data is avaliable [here.](https://www.amazon.fr/Waveshare-Raspberry-Camera-Fisheye-Raspberry-pi/dp/B00RMV53Z2/)

The images were labeled as follows:
![alt text](https://github.com/Axionable/AxionautV2/blob/master/Docs/labels.png)

If the labels are consistent, your training and validation accuracy should increase importantly. It is because the labels respect the inner structure of your data. If you get that, you are in a good path :)


## Data augmentation
Augmenting your data is a good way to improve the generalisation capabilities of your model. There is a lot of information about this on the Internet. We took some functions and we added some more to easily allow you to generate new data. All functions are available in the`functions.py` script.

![alt text](https://github.com/Axionable/AxionautV2/blob/master/Docs/augmentation.png)


## Data preparation
Before training your model, it is necessary to balance the number of examples per class. Doing so, you will avoid biases, ensuring a smooth driving behavior in almost all road conditions. In general, we try to preserve a 1:1 ratio between curves and straights examples, we also give less frequency to very hard turns. It is recommended to test the car after training a model with balanced data, see if it learned well to go straight, turn left and right. If there is a task it does not perform well, just add more data of that class and train again. After a few iterations your car will be ready to run on any track :).

A good way to see the global distribution of your data is the histogram. In the `data preparation.ipyn` jupyter notebook, you will find some code examples to balance your own dataset using [Pandas.](https://pandas.pydata.org)

![alt text](https://github.com/Axionable/AxionautV2/blob/master/Docs/histograms.png)


## The model
The proposed architecture is a slightly modified version of the PilotNet published by [Nvidia](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). This architecture is powerful to modelize all possible driving situations while simple enough to run on the raspberry pi 3 B+. [Dropout](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf) of 10% was added on two classifier layers to avoid [overfitting](https://en.wikipedia.org/wiki/Overfitting).


## Code Exemple

The `python train.py` script allows to train a driving model using the aforementioned architecture and dataset. By default, the script will save the best model snapshot after each epoch if the validation loss decreased. Normally, the default settings works well, however you can freely modify the settings according the following examples:

<strong>Start training with default settings:</strong>
`python train.py`


### List of parameters:

1. <strong>--augmentation:</strong> Use data augmentation functions, default = True.
2. <strong>--val_split:</strong> Validation split, default 0.2.
3. <strong>--epochs:</strong> Number of training epochs, default 100.
4. <strong>--batch_size:</strong> Size of batch, default 64 images.
5. <strong>--early_stop:</strong> Use early stop, default True.
6. <strong>--patience:</strong> Maximum number of epochs without loss improvement, default 5.


<strong>To train your own driving model:</strong>
`python train.py --augmentation True --epochs 10 --batch_size 128 --patience 5`

Feel free to explore and set your prefered training hyperparameters!


## Installation

<strong>Clone repository to your laptop:</strong>
`git clone https://github.com/Axionable/AxionautV2`

<strong>Install required libraries:</strong>
`pip install -r requirements.txt`


## Contribute

Axionaut is totally free and open for everyone to use, please feel free to contribute!


## About Axionable

Axionable is a leading Data Science and Data Consulting firm based on Paris, France. For knowing more about our projects and careers please visit our [website](https://www.axionable.com). Follow us on [Twitter](https://twitter.com/AxionableData).

