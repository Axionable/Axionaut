import cv2
import numpy as np
import random
import skimage.exposure as sk
from tqdm import tqdm


def normalisation(image):
    '''Return image centered around 0 with +- 0.5.

    image: Image to transform in Numpy array.

    '''
    return image/255.-.5


def horizontal_flip(img, angle):
    """Horizontal image flipping and angle correction.

    Img: Input image to transform in Numpy array.
    Angle: Corresponding label. Must be a 5-elements Numpy Array.
    """    
    return np.fliplr(img), np.flipud(angle)


def augment_brightness_camera_images(image, angle):
    '''Random bright augmentation (both darker and brighter).
    
    Returns:
    Transformed image and label.
    '''
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1, angle


def add_random_shadow(image, angle):
    '''Add random dark shadows to a given image.

    Returns:
    Transformed image and label.
    '''
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image, angle


def night_effect(img,  label, vmin=185, vmax=195):
    """Change road color to black simulating night road.

    Returns
    Transformed image and label.
    """
    limit = random.uniform(vmin,vmax)
    low_limit = 146 
    int_img = sk.rescale_intensity(img, in_range=(low_limit,limit), out_range='dtype')
    
    return int_img, label


def adjust_gamma_dark(image, label, min_=0.7, max_=0.8):
    '''Gamma correction to generate darker images.

    Image: Image in Numpy format (90,250,3)
    Label: Corresponding label of the image.
    Min: Minimum gamma value (the lower the darker)
    Max: Maximum gamma value (the higher the brigther) 
    Return:
    Transformed image and label.
    '''
    # build a lookup table mapping the pixel values [0, 255] to
    gamma = random.uniform(min_,max_)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table), label



def generate_brightness(X, Y, proportion=0.25):
    '''Image augmentation changing the intensity randomly.

    X: Numpy array of training images in channels-last format.
    Y: Labels (array of 5 elements).
    proportion: percentage of elements to be generated (25% of total by default).
    Returns
    X_aug: Numpy tensor of images according to the desired proportion.
    Y_aug: Numpy tensor of labels for each image.
    '''
    # Generate a random selection of indexes
    indexes = random.sample(range(0, X.shape[0]), int(X.shape[0]*proportion))
    
    X_aug = []
    Y_aug = []
    for index in tqdm(indexes):
        # Apply the desired transformation
        im, angle = augment_brightness_camera_images(X[index], Y[index])
        Y_aug.append(angle)
        X_aug.append(im)

    return X_aug, Y_aug


def generate_low_gamma(X, Y, proportion=0.25, min_=0.7, max_=0.8):
    '''Image augmentation by lowering the gamma (darker images).

    X: Numpy array of training images in channels-last format.
    Y: Labels (array of 5 elements).
    proportion: percentage of elements to be generated (25% of total by default).
    Returns
    X_aug: Numpy tensor of images according to the desired proportion.
    Y_aug: Numpy tensor of labels for each image.
    '''
    # Generate a random selection of indexes
    indexes = random.sample(range(0, X.shape[0]), int(X.shape[0]*proportion))
    
    X_aug = []
    Y_aug = []
    for index in tqdm(indexes):
        # Apply the desired transformation
        im, angle = adjust_gamma_dark(X[index], Y[index], min_, max_)
        Y_aug.append(angle)
        X_aug.append(im)

    return X_aug, Y_aug


def generate_night_effect(X, Y, proportion=0.25):
    '''Image augmentation with night effect (black track).

    X: Numpy array of training images in channels-last format.
    Y: Labels (array of 5 elements).
    proportion: percentage of elements to be generated (25% of total by default).
    Returns
    X_aug: Numpy tensor of images according to the desired proportion.
    Y_aug: Numpy tensor of labels for each image.
    '''
    # Generate a random selection of indexes
    indexes = random.sample(range(0, X.shape[0]), int(X.shape[0]*proportion))
    
    X_aug = []
    Y_aug = []
    for index in tqdm(indexes):
        # Apply the desired transformation
        im, angle = night_effect(X[index], Y[index])
        Y_aug.append(angle)
        X_aug.append(im)

    return X_aug, Y_aug


def generate_horizontal_flip(X, Y, proportion=0.25):
    '''Image augmentation by lowering the gamma (darker images).

    X: Numpy array of training images in channels-last format.
    Y: Labels MUST be a 5-elements array.
    proportion: percentage of elements to be generated (25% of total by default).
    Returns
    X_aug: Numpy tensor of images according to the desired proportion.
    Y_aug: Numpy tensor of labels for each image.
    '''
    # Generate a random selection of indexes
    indexes = random.sample(range(0, X.shape[0]), int(X.shape[0]*proportion))
    
    X_aug = []
    Y_aug = []
    for index in tqdm(indexes):
        # Apply the desired transformation
        im, angle = horizontal_flip(X[index], Y[index])
        Y_aug.append(angle)
        X_aug.append(im)

    return X_aug, Y_aug


def generate_random_shadows(X, Y, proportion=0.25):
    '''Image augmentation by lowering the gamma (darker images).

    X: Numpy array of training images in channels-last format.
    Y: Labels MUST be a 5-elements array.
    proportion: percentage of elements to be generated (25% of total by default).
    Returns
    X_aug: Numpy tensor of images according to the desired proportion.
    Y_aug: Numpy tensor of labels for each image.
    '''
    # Generate a random selection of indexes
    indexes = random.sample(range(0, X.shape[0]), int(X.shape[0]*proportion))
    
    X_aug = []
    Y_aug = []
    for index in tqdm(indexes):
        # Apply the desired transformation
        im, angle = add_random_shadow(X[index], Y[index])
        Y_aug.append(angle)
        X_aug.append(im)

    return X_aug, Y_aug


def generate_chained_transformations(X, Y, proportion=0.25):
    '''Image augmentation applying brightness + shadow + random flipping.

    X: Numpy array of training images in channels-last format.
    Y: Labels MUST be a 5-elements array.
    proportion: percentage of elements to be generated (25% of total by default).
    Returns
    X_aug: Numpy tensor of images according to the desired proportion.
    Y_aug: Numpy tensor of labels for each image.
    '''
    # Generate a random selection of indexes
    indexes = random.sample(range(0, X.shape[0]), int(X.shape[0]*proportion))
    
    X_aug = []
    Y_aug = []
    for index in tqdm(indexes):
        # Apply the desired transformation
        im, angle = augment_brightness_camera_images(X[index], Y[index])
        im, angle =  add_random_shadow(im, angle)
    
        if index%2==0:
            im, angle = horizontal_flip(im, angle)
        Y_aug.append(angle)
        X_aug.append(im)
    
    return X_aug, Y_aug



