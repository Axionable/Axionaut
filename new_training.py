import numpy as np
from functions import *
from architectures import *
import keras
from keras.models import load_model
from sklearn.model_selection import train_test_split



# Load out previous pre-trained model
model = load_model('Models/checkpoint_sun_FULL.hdf5')

# Set path to save trained model
model_path = 'Models/'

# Load training data
# Get training data from the Axionable track
X_axio = np.load('Datasets/axionable_data/X_train_axio.npy')
Y_axio = np.load('Datasets/axionable_data/Y_train_axio.npy')
print('Axionable data Loaded. Shape = ', np.shape(X_axio))

# Data augmentation of the dataset / Adjust the proportion of each transformation you want to apply.
# Use augmentation by default
augmentation = True

if augmentation:    
    print('Augmenting data... Wait...')
    
    # Data augmentation 35% of random sunglow.
    X_sun, Y_sun = generate_random_sunglow(X_axio, Y_axio, proportion=0.35)
    
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
                                X_flip, X_shadow, X_chain, X_sun))

    Y_axio = np.concatenate((Y_axio, Y_bright, Y_night, 
                                Y_flip, Y_shadow, Y_chain, Y_sun)).astype('float32')

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

# New double Chicane
X_chicane2 = np.load('Datasets/new/x_chicane.npy')
Y_chicane2 = np.load('Datasets/new/y_chicane.npy')
print('Ironcar new track chicane Loaded. Shape = ', np.shape(X_chicane))


# Data augmentation of data taken on the IronCar track 
#/ Adjust the proportion of each transformation you want to apply.

if augmentation:
    print('Augmenting data... Wait...')
    
    # Augmentation of old"Double Chicane" Proportion=1. Duplicate the original data (aprox 1500 images)
    X_chicane_sun, Y_chicane_sun = generate_random_sunglow(X_chicane, Y_chicane, proportion=0.7)
    X_chicane_aug, Y_chicane_aug = generate_brightness(X_chicane, Y_chicane, proportion=0.7)
    
    # NEW CHICANA
    X_chicane_aug2, Y_chicane_aug2 = generate_random_sunglow(X_chicane2, Y_chicane2, proportion=1)
    X_chicane_aug3, Y_chicane_aug3 = generate_brightness(X_chicane2, Y_chicane2, proportion=1)
    
    # Irocar Balanced Dataset
    # 25% of random bright transformations 
    X_bright_iron, Y_bright_iron = generate_brightness(X_iron, Y_iron, proportion=0.25)
    # 25% of lo gamma transformations (darker images)
    X_gamma_iron, Y_gamma_iron = generate_low_gamma(X_iron, Y_iron, proportion=0.25, min_=0.7, max_=0.8)
    
    # 25% of lo gamma transformations (darker images)
    X_sun_iron, Y_sun_iron = generate_random_sunglow(X_iron, Y_iron, proportion=0.4)

    # Concatenating IronCar dataset with the transformations.
    X_iron = np.concatenate((X_iron, X_chicane_aug, X_chicane_sun, X_sun_iron,
                             X_bright_iron,X_gamma_iron, X_chicane_aug2, 
                             X_chicane_aug3, X_chicane2))

    Y_iron = np.concatenate((Y_iron, Y_chicane_aug, Y_chicane_sun, Y_sun_iron,
                             Y_bright_iron, Y_gamma_iron, Y_chicane_aug2, 
                             Y_chicane_aug3, Y_chicane2)).astype('float32')

    print('Ironcar data after augmentation. Shape = ', np.shape(X_iron))

# Concatenate both augmented datasets in a single one
X = np.concatenate((X_axio, X_iron))
Y = np.concatenate((Y_axio, Y_iron))
print('Total X shape after augmentation = ', np.shape(X))

del X_iron, Y_iron, X_axio, Y_axio, X_chicane2, Y_chicane2, X_gamma_iron
del X_chicane_aug, X_chicane_sun, X_sun_iron, X_bright_iron, X_chicane_aug2

# Shuffling examples / Create train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

del X, Y


# Train model
model_name = model_path + 'checkpoint_sun_FULL_TL.hdf5'
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
                                           patience=5, 
                                           verbose=1, 
                                           mode='auto')

callbacks_list = [save_best, early_stop]

#early_stop = True

#if early_stop:
#    callbacks_list.append(early_stop)

hist = model.fit(
                X_train, 
                y_train,
                nb_epoch=1,
                batch_size=64, 
                verbose=1, 
                validation_data=(X_test, y_test),
                callbacks= callbacks_list, #[save_best], 
                shuffle=True)








