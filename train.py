import numpy as np 
import keras
from sklearn.model_selection import train_test_split
from functions import *
from architectures import *
import argparse
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='Axionaut training')
parser.add_argument('--augmentation', default=True)
parser.add_argument('--val_split', default=0.2)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--early_stop', default=True)
parser.add_argument('--patience', default=5, type=int)

args = parser.parse_args()

# Model path
model_path = 'Models/'

# Get training data from the Axionable track
X_axio = np.load('Datasets/axionable_data/X_train_axio.npy')
Y_axio = np.load('Datasets/axionable_data/Y_train_axio.npy')
print('Axionable data Loaded. Shape = ', np.shape(X_axio))

# Data augmentation of the dataset / Adjust the proportion of each transformation you want to apply.
if args.augmentation:
    print('Augmenting data... Wait...')
    # Data augmentation 25% of random brightness.
    X_bright, Y_bright = generate_brightness(X_axio, Y_axio, proportion=0.25)
    # Data augmentation 25% of night effect.
    X_night, Y_night = generate_night_effect(X_axio, Y_axio, proportion=0.25)
    # Data augmentation 25% of horizontal flipping.
    X_flip, Y_flip = generate_horizontal_flip(X_axio, Y_axio, proportion=0.25)
    # Data augmentation 25% of random shadows.
    X_shadow, Y_shadow = generate_random_shadows(X_axio, Y_axio, proportion=0.25)
    # Data augmentation 25% of chained tranformations (bright + shadows + flip).
    X_chain, Y_chain = generate_chained_transformations(X_axio, Y_axio, proportion=0.25)

    # Concatenating Axionable dataset with the transformations.
    X_axio = np.concatenate((X_axio, X_bright, X_night,
                                X_flip, X_shadow, X_chain))

    Y_axio = np.concatenate((Y_axio, Y_bright, Y_night, 
                                Y_flip, Y_shadow, Y_chain)).astype('float32')

    print('Axionable data after augmentation. Shape = ', np.shape(X_axio))

# Get training data from IronCar track
# New track - Double chicane
X_chicane = np.load('Datasets/ironcar_data/new_track/x_chicane.npy')
Y_chicane = np.load('Datasets/ironcar_data/new_track/y_chicane.npy')
print('Ironcar new track chicane Loaded. Shape = ', np.shape(X_chicane))

# Old track - Balanced dataset
X_iron = np.load('Datasets/ironcar_data/old_track/balanced_iron_X.npy')
Y_iron = np.load('Datasets/ironcar_data/old_track/balanced_iron_Y.npy')
print('Ironcar old track data Loaded. Shape = ', np.shape(X_iron))

# Data augmentation of data taken on the IronCar track / Adjust the proportion of each transformation you want to apply.
if args.augmentation:
    print('Augmenting data... Wait...')
    # Augmentation of the "Double Chicane" Proportion=1. Duplicate the original data (aprox 1500 images)
    X_chicane_aug, Y_chicane_aug = generate_brightness(X_chicane, Y_chicane, proportion=1)
    # Irocar Balanced Dataset
    # 25% of random bright transformations 
    X_bright_iron, Y_bright_iron = generate_brightness(X_iron, Y_iron, proportion=0.25)
    # 25% of lo gamma transformations (darker images)
    X_gamma_iron, Y_gamma_iron = generate_low_gamma(X_iron, Y_iron, proportion=0.25, min_=0.7, max_=0.8)

    # Concatenating IronCar dataset with the transformations.
    X_iron = np.concatenate((X_iron, X_chicane_aug, 
                                   X_bright_iron,X_gamma_iron))

    Y_iron = np.concatenate((Y_iron, Y_chicane_aug, 
                                   Y_bright_iron, Y_gamma_iron)).astype('float32')

    print('Ironcar data after augmentation. Shape = ', np.shape(X_iron))

# Concatenate both augmented datasets in a single one
X = np.concatenate((X_axio, X_iron))
Y = np.concatenate((Y_axio, Y_iron))

# Perform train/validation split. We do not use test set.
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=args.val_split, random_state=12)

# Create autopilot model from architectures and print summary
model =  model_categorical(input_size=(90,250,3), dropout=0.1)
model.summary()

# Train model
model_name = model_path + 'checkpoint_model.hdf5'
min_delta=.0005

#checkpoint to save model after each epoch
save_best = keras.callbacks.ModelCheckpoint(model_name, 
                                            monitor='val_loss', 
                                            verbose=1, 
                                            save_best_only=True, 
                                            mode='min')

#stop training if the validation error stops improving.
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                           min_delta=min_delta, 
                                           patience=args.patience, 
                                           verbose=1, 
                                           mode='auto')

callbacks_list = [save_best]

if args.early_stop:
    callbacks_list.append(early_stop)

hist = model.fit(
                X_train, 
                Y_train,
                nb_epoch=args.epochs,
                batch_size=args.batch_size, 
                verbose=1, 
                validation_data=(X_val, Y_val),
                callbacks=callbacks_list,
                shuffle=True)


model.save('final_model.hdf5')

