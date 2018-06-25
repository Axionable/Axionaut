## Axionaut - WORK IN PROGRESS
Mini autonomous RC vehicle for AI enthusiasts.

## Introduction
Axionaut provides a straightforward approach to prototype mini RC autonomous vehicles. 

The complete bill of materials is avaliable here:
https://www.axionable.com/axionaut-termine-1er-de-la-deuxieme-course-iron-car/

Axionaut is intended for rapid experimentation, use the built-in Deep Learning architectures and start driving!


## Code style
PEP 8 -- Style Guide for Python Code.


## Screenshot
![alt text](https://www.axionable.com/wp-content/uploads/2018/02/axionautV1.png)


## Tech/framework used

<b>Built using:</b>
- [TensorFlow](https://www.tensorflow.org)
- [Keras](https://keras.io)


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

