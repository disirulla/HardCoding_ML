#Author: Alluri L S V Siddhartha Varma
#Date: 18-12-2020


import numpy as np

def convolution(image, kernel_size = (3,3), padding = 0, pad_with = 0, conv_type = 'valid'):
    #TODO: Add Strides
    '''
        Convolution of an Image: Convolution is kind of cross-corelation operation on images with a matrix called filter or kernel./n
        It is very prominent technique used in Deep Learning for extracting features from images. Basic block of Convolutional neural network./n
        
        image: input an image
        kernel_size: size of the kernel, default is (3,3)
        padding: no. layers to be padded, default = 0
        pad_with: the integer to pad layers with, default = 0
        convolution type = either 'valid' or 'same', well known premise in convolution of images
    
    '''
    kernel = np.random.randn(kernel_size[0],kernel_size[1])
    print(kernel)
    convolved_image = np.zeros((image.shape[0]+(2*padding)-kernel_size[0]+1,image.shape[1]+(2*padding)-kernel_size[1]+1))
    if(conv_type == 'same'):
        padding = (kernel_size[0]-1)/2
        
    if (padding != 0):
        image_padded = np.zeros(image.shape[0]+padding,image.shape[0]+padding)
        if (pad_with != 0):
            image_padded = image_padded + pad_with
        pos = int(padding/2)
        image_padded[pos:-pos, pos:-pos] = image
        image = image_padded
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                convolved_image[i,j] = np.sum(image[i:i+kernel_size[0], j:j+kernel_size[1]] * kernel)
                
    else:
        for i in range(image.shape[0] - kernel_size[0]+1):
            for j in range(image.shape[1] - kernel_size[1]+1):
                convolved_image[i,j] = np.sum(image[i:i+kernel_size[0], j:j+kernel_size[1]] * kernel)
        
    return convolved_image
    
