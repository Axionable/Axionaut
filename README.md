## AxionautV2 - DIY Robocar Tricks and Hacks
The Axionaut is the winner of the second [IronCar France](http://ironcar.org) robocar competition. The following repository contains all the tips and tricks we used to win. The provided autopilot model will work seameless with Vincent's [IronCar](https://github.com/vinzeebreak/ironcar) framework.


## Introduction
The Axionaut is based on a 4x4 Radio Controlled (RC) monster truck chasis, the complete bill of materials is avaliable here:
https://www.axionable.com/axionaut-termine-1er-de-la-deuxieme-course-iron-car/

The following code and documentation will guide you through the best practices and tricks to train the car. All strategies were defined both searching on the Internet and by experimention. So, are you ready to win your next race?

## Screenshot
![alt text](https://www.axionable.com/wp-content/uploads/2018/02/axionautV1.png)


## Code style
PEP 8 -- Style Guide for Python Code.


## Tech/framework used

<b>Built using:</b>
- [TensorFlow](https://www.tensorflow.org)
- [Keras](https://keras.io)

## Finding the right camera position
Make sure to put the camera such that the three lanes are always visible. Also, crop the original image to keep only the track. The position and crop level depends on your vehicle and the camera you are using. In our case the image was cropped to [90, 250, 3] and the camera was placed at 20 cm from the floor. We recommend the use of [eyefish cameras.](https://www.amazon.fr/Waveshare-Raspberry-Camera-Fisheye-Raspberry-pi/dp/B00RMV53Z2/ref=pd_cp_23_3?_encoding=UTF8&psc=1&refRID=7JBTZQTNFRVC34PY6J4X)

It is normally easy to know when the camera is not in the right position. Here an usefull example, make sure to check your training data before training your model. Once you found a good fit, fix the camera.
![alt text](https://github.com/Axionable/AxionautV2/blob/master/Docs/camera_adj.png)


## Data preparation
Before training your model, it is necessary to balance the number of examples per class. Doing so, you will avoid biases assuring a smooth driving behavior in almost all road conditions. In general, we try to preserve a 1:1 ratio between curves and straights examples, we also give less frequency to very hard turns. It is recommended to test the car after training a model with balanced data, see if it learned well to go straight, turn left and right. If there is a task it does not perform well, just add more data of that class and train again. After a few iterations your car will be ready to run on any track :).

A good way to see the global distribution of your data is the histogram. In the `data preparation.ipyn` jupyter notebook, you will find some code examples to balance your own dataset using [Pandas.](https://pandas.pydata.org)

![alt text](https://github.com/Axionable/AxionautV2/blob/master/Docs/histograms.png)



## Features

1. <strong>Autonomous drive mode:</strong> Real-time autopilot using Deep Learning models.
2. <strong>Data recording:</strong> Real-time data recording from the car.
3. <strong>Training mode:</strong> Build and train your own driving models from scratch or using transfer learning.
4. <strong>Free ride:</strong> Enjoy driving your RC car on the free ride mode.


## API

Create a new vehicle and set it to self-driving mode is extremely easy:

	#Load self-driving pre trained model
    model, graph = load_autopilot('autopilot.hdf5')

    # Create Axionaut car with default settings
    axionaut = vehicles.Axionaut()

    # Configure PDW control commands as default
    axionaut.commands = get_commands(path=None, default=True)

    # Test camera position
    axionaut.camera_test()

    # Set vehicle to auto pilot mode 
    axionaut.autopilot(model, graph)

    # Start car   
    axionaut.start()

## Code Exemple

The following commands are avaliable when using the main.py example:

<strong>Start vehicle on self-driving mode:</strong>
`python main.py --mode self_driving`

<strong>Start on recording mode:</strong>
`python main.py --mode record`

<strong>Start on free ride mode:</strong>
`python main.py --mode free`

<strong>To train your own driving model:</strong>
`python main.py --mode train --architecture ConvNets --epochs 100 --batch_size 300 --optimizer Adam`

Feel free to explore and set your prefered training hyperparameters!


## Installation
### Raspberry side:
<strong>Clone repository to your Raspberry Pi:</strong>
`git clone https://github.com/Axionable/AxionautV1`

<strong>Install packages:</strong>
`pip install -r requirements.txt`

### Computer side:
<strong>Clone repository to your laptop:</strong>
`git clone https://github.com/Axionable/AxionautV1`

<strong>Install packages:</strong>
`pip install -r laptop_requirements.txt`


## Status

Axionaut is currently under active developement.

## Contribute

Axionaut is totally free and open for everyone to use, please feel free to contribute!

## Credits
Special thanks to IronCar France and Vincent Houlbrèque's great repository:
- [vinzeebreak/ironcar](https://github.com/vinzeebreak/ironcar)



## About Axionable

Axionable is a leading Data Science and Data Consulting firm based on Paris, France. For knowing more about our projects and careers please visit our [website](https://www.axionable.com). Follow us on [Twitter](https://twitter.com/AxionableData).

