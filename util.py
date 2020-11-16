#import libraries
import cv2 as cv
import os
from skimage import io
import skimage
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import scipy.misc
from mpl_toolkits.mplot3d import Axes3D

org_images  = []

def Load_images(path):
    #load files from directory
    for filename in os.listdir(path):
        #Read images using opencv
        img = cv.imread(os.path.join(path,filename))
        if img is not None:
            org_images.append(img)
  
    return org_images

#Task1
gray_images = []

def display_images(imgs):
    #define figure size
    fig = plt.figure(figsize=(15, 15))
    itr = 1
    
    #Display images in 1 row
    for i in imgs:
        fig.add_subplot(1, 9, itr)
        fig.tight_layout()
        itr = itr + 1
        plt.imshow(i)
        
def convert_to_gray(in_imgs):
    for image in in_imgs:
        #convert images into grayscale
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        #save images in list
        gray_images.append(gray)
    
    return gray_images

#Task2

green_images = []
red_images   = []
blue_images  = []

def rgbExclusion(s_images):
    
    iteration = 0
    
    dim, axis = plt.subplots(8,4,figsize=(15,15))
    for pic in s_images:
        if iteration < 8:
            #split the color channels of images
            b_,g_,r_ = cv.split(pic)
            
            kernel = np.zeros_like(b_)

            #construct seperate channel images i.e. R, G, B
            blue_images.append(cv.merge([b_,kernel,kernel]))
            green_images.append(cv.merge([kernel,g_,kernel]))
            red_images.append(cv.merge([kernel,kernel,r_]))

            #display images
            axis[iteration,0].imshow(org_images[iteration])
            axis[iteration,2].imshow(red_images[iteration])
            axis[iteration,1].imshow(green_images[iteration])
            axis[iteration,3].imshow(blue_images[iteration]) 

            iteration = iteration + 1
            
#Task3
def HistoEqualization(input_img):
    for image__ in input_img:
        src = cv.cvtColor(image__, cv.COLOR_BGR2GRAY)
        #equalize the color channels of image
        equ = cv.equalizeHist(src)
        
        #display images i.e. source and equalized
        fig, ax = plt.subplots(1,2)
        ax[0].title.set_text('Before')
        ax[0].imshow(src)

        ax[1].title.set_text('After')
        ax[1].imshow(equ)
        
        #display histograms
        fig, bx = plt.subplots(1,2)
        bx[0].hist(src.flatten(),256,[0,255], color = 'black')
        bx[1].hist(equ.flatten(),256,[0,255], color = 'black')
plt.show()


#Task4

def Self_Convolution_Operation(image, kernel):
    kernel = np.flipud(np.fliplr(kernel))    # Flip the kernel
    output = np.zeros_like(image)            # convolution output
    # Add zero padding to the input image
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))   
    image_padded[1:-1, 1:-1] = image
    
    # Loop over every pixel of the image and implement convolution operation (element wise multiplication and summation). 
    # You can use two loops. The result is stored in the variable output.
    
    for x in range(image.shape[0]):     # Loop over every pixel of the image
        for y in range(image.shape[1]):
            # element-wise multiplication and summation 
            output[x,y]=(kernel*image_padded[x:x+3,y:y+3]).sum()
    return output

def show_sharpen(image_sharpen, sharpened):
    fig, ax = plt.subplots(1,2)
    ax[0].title.set_text('Sharp (From Scratch)')
    ax[0].imshow(image_sharpen,cmap=plt.cm.gray)

    ax[1].title.set_text('Sharp (Built-in)')
    ax[1].imshow(sharpened,cmap=plt.cm.gray)

def show_blur(image_blurred,blur):
    fig, ax = plt.subplots(1,2)
    ax[0].title.set_text('Blur (From Scratch)')
    ax[0].imshow(image_blurred,cmap=plt.cm.gray)

    ax[1].title.set_text('Blur (Built-in)')
    ax[1].imshow(blur, cmap=plt.cm.gray)

#Task 5-1

def show_box_filter(image,f):
    fig, ax = plt.subplots(1,2)
    ax[0].title.set_text('Original')
    ax[0].imshow(image,cmap=plt.cm.gray)

    ax[1].title.set_text('Box Filter')
    ax[1].imshow(f, cmap=plt.cm.gray)
    
#Task 5-2

def gaussian_filter(image):
    #built-in gaussian blur image function
    gaussian_sig_4 = cv.GaussianBlur(image, (5,5),sigmaX=4)
    gaussian_sig_6 = cv.GaussianBlur(image, (5,5),sigmaX=6)
    gaussian_sig_8 = cv.GaussianBlur(image, (5,5),sigmaX=8)

    fig, ax = plt.subplots(1,4, figsize = (15,15))

    ax[0].title.set_text('Original')
    ax[0].imshow(image,cmap=plt.cm.gray)

    ax[1].title.set_text('Sigma = 4')
    ax[1].imshow(gaussian_sig_4, cmap=plt.cm.gray)

    ax[2].title.set_text('Sigma = 6')
    ax[2].imshow(gaussian_sig_6,cmap=plt.cm.gray)

    ax[3].title.set_text('Sigma = 8')
    ax[3].imshow(gaussian_sig_8, cmap=plt.cm.gray)
    
#Task 5-3

def plotnoise(img, mode):
    gimg = skimage.util.random_noise(img, mode = mode)
    return gimg

def salt_pepper(img, SNR):
    img_ = img.copy()
    c, h, w = img_.shape
    mask = np.random.choice((0, 1, 2), size=(1, h, w), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
    Mask = np.repeat(mask, c, axis=0) # Copy by channel to have the same shape as img
    Img_[mask == 1] = 255 # salt noise
    Img_[mask == 2] = 0 # 

    return img_

def show_noise_images(gaussian_image, salt_image, pepper_image):
    fig, ax = plt.subplots(1,3, figsize = (15,15))

    ax[0].title.set_text('Gaussian')
    ax[0].imshow(gaussian_image)

    ax[1].title.set_text('salt')
    ax[1].imshow(salt_image)

    ax[2].title.set_text('pepper')
    ax[2].imshow(pepper_image)
    
#Task 5-4

def gaussian_median_show(image, gaussian_filter, median_filter):
    fig, ax = plt.subplots(1,3, figsize = (15,15))
    ax[0].title.set_text('Original')
    ax[0].imshow(image)

    ax[1].title.set_text('Gaussian')
    ax[1].imshow(gaussian_filter)

    ax[2].title.set_text('Median')
    ax[2].imshow(median_filter)

#Task 5-5

def mesh_plot(img, title):
    img_ = img

    # create the x and y coordinate arrays (here we just use pixel indices)
    xx, yy = np.mgrid[0:img_.shape[0], 0:img_.shape[1]]

    # create the figure
    fig = plt.figure(figsize=(5,5))
    ax = fig.gca(projection='3d')
    ax.plot_surface(xx, yy, img_ ,rstride=1, cstride=1, cmap=plt.cm.jet,linewidth=0)
    ax.title.set_text(title)

    # show it
    plt.show()
    
#Task 6-1
def sobel_show(img, img_SobelX, img_SobelY):
    fig, ax = plt.subplots(1,3, figsize = (15,15))
    ax[0].title.set_text('Original')
    ax[0].imshow(img, cmap=plt.cm.gray)

    ax[1].title.set_text('Sobel X')
    ax[1].imshow(img_SobelX, cmap=plt.cm.gray)

    ax[2].title.set_text('Sobel Y')
    ax[2].imshow(img_SobelY, cmap=plt.cm.gray)

#Task 6-2
def laplacian_show(gray, img, laplacian):
    fig, ax = plt.subplots(1,3, figsize = (15,15))
    ax[0].title.set_text('Gray')
    ax[0].imshow(gray, cmap=plt.cm.gray)

    ax[1].title.set_text('Gaussian')
    ax[1].imshow(img, cmap=plt.cm.gray)

    ax[2].title.set_text('Laplacian')
    ax[2].imshow(laplacian, cmap=plt.cm.gray)
    
#Task 6-3
def canny_show(img, edges):
    fig, ax = plt.subplots(1,2, figsize = (15,15))
    ax[0].title.set_text('Original')
    ax[0].imshow(img,cmap=plt.cm.gray)

    ax[1].title.set_text('Canny')
    ax[1].imshow(edges,cmap=plt.cm.gray)
    
#Bonus Task

def video_canny():
    
    stream = cv.VideoCapture('video.mp4')

    while True:
        # To get connectivity status and frames from video
        ret, frame = stream.read()  
        
        frame_ = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        edge = cv.Canny(frame, 100, 200)

        #resize the window
        cv.resizeWindow('Original Video', 200, 200)
        
        #display the canny window
        cv.imshow('Canny Edge', edge)
        
        #press q to exit.
        if cv.waitKey(20) == ord('q'):  
            break