import cv2
import numpy as np
import matplotlib.pyplot as plt

### Exercise 1

img3 = cv2.imread('image1.jpg', 0).astype('float')
h1 = np.ones((5,5)) / 25;
img4 = cv2.filter2D(img3,-1,h1)

# a)
img4_indexed = img4[2:-2,2:-2]

plt.figure()
plt.imshow(img4_indexed,cmap='gray')
plt.title('Filtered img3 without borders, indexed')

# b)

from scipy.signal import convolve2d

img4_conv2 = convolve2d(img3.astype('float'),h1,mode='valid')

plt.figure()
plt.imshow(img4_conv2,cmap='gray')
plt.title('Filtered img3 without borders, convolve2d')

#plt.show()

# Just to check if a) and b) gives the same answer:
print("Max absolute error between index and filtered using valid: %g"\
      %np.max(np.abs(img4_conv2 - img4_indexed)))

### Exercise 2

hist_np,bins = np.histogram(img3.ravel(), bins=256,range=(0,256))

def iimhist(img):
    h = np.zeros(256)
    for i in range(256):
        h[i] = np.sum(img == i)
        # img == i makes a boolean matrix. By summing over the matrix,
        # all entries being False evaluates to 0, and all entries being True
        # to 1.
    return h

hist_ = iimhist(img3)

# Too check is the implementation is correct compared to np.histogram
print("Max absolute difference between np.histogram and iimhist: %g"%np.max(np.abs(hist_np - hist_)))

plt.figure()
plt.bar(bins[:-1], hist_np) #take the whole bin array except for the last element,
                            #this is equal to 256.
plt.bar(bins[:-1], hist_)
plt.title('Histogram using numpy')
plt.legend(['numpy', 'iimhist'])

#plt.show()


### Exercise 3

# a)
img2 = cv2.imread('image2.jpg', 0)

img2_size = img2.shape

thresholded1 = np.zeros(img2_size) # Make a matrix of the same size as img2
thresholded1[img2 > 100] = 1

thresholded2 = np.zeros(img2_size)
thresholded2[img2 < 100] = 1

thresholded3 = np.zeros(img2_size)
thresholded3[img2 >= 120] = 1

thresholded4 = np.zeros(img2_size)
thresholded4[img2 <= 120] = 1

# Make a figure with several plots iwhtin a window
plt.figure()

plt.subplot(2,2,1)
plt.imshow(thresholded1,cmap='gray')
plt.title("Pixel values from img2 greater than 100")

plt.subplot(2,2,2)
plt.imshow(thresholded2,cmap='gray')
plt.title("Pixel values from img2 less than 100")

plt.subplot(2,2,3)
plt.imshow(thresholded3,cmap='gray')
plt.title("Pixel values from img2 greater or equal to 120")

plt.subplot(2,2,4)
plt.imshow(thresholded4,cmap='gray')
plt.title("Pixel values from img2 less or equal to 120")

#plt.show()

# b)
thr_otsu,img2_otsu = cv2.threshold(img2,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

plt.figure()
plt.imshow(img2_otsu,cmap='gray')
plt.title('img2 thresholded with Otsu`s method')

#plt.show()

# c)
plt.figure()

plt.subplot(2,2,1)
plt.imshow(np.abs(thresholded1-img2_otsu),cmap='gray')
plt.title("Difference between thresholded1 and \n thresholded image using Otsu`s method")

plt.subplot(2,2,2)
plt.imshow(np.abs(thresholded2-img2_otsu),cmap='gray')
plt.title("Difference between thresholded2 and \n thresholded image using Otsu`s method")

plt.subplot(2,2,3)
plt.imshow(np.abs(thresholded3-img2_otsu),cmap='gray')
plt.title("Difference between thresholded3 and \n thresholded image using Otsu`s method")

plt.subplot(2,2,4)
plt.imshow(np.abs(thresholded4-img2_otsu),cmap='gray')
plt.title("Difference between thresholded4 and \n thresholded image using Otsu`s method")

#plt.show()


### Exercise 4

h2x = np.array([[-1,-2,-1],\
                [0,0,0],\
                [1,2,1]])

h2y = np.array([[-1,0,1],\
                [-2,0,2],\
                [-1,0,1]])

resX = cv2.filter2D(img3,-1, h2x)
resY = cv2.filter2D(img3,-1, h2y)
resXY = np.sqrt(resX**2 + resY**2)

resXY_norm = cv2.normalize(resXY, None, 0, 255, cv2.NORM_MINMAX)

print("max(resXY_norm) = %g"%np.max(resXY_norm))
print("min(resXY_norm) = %g"%np.min(resXY_norm))

# One could also do the normalization 'manually'.
# The normalization could be seen as a linear transform of the pixel
# intensities: normalized_img = a*img + b
# To determine the coefficients a and b, the following equations can
# be solved:
#
# 0 = a*min(img) + b
# 255 = a*max(img) + b
#
# which yields
# a = 255/( max(img) - min(img) )
# b = -a*min(img)
#   = -( 255 * min(img) )/( max(img) - min(img) )
#
# and gives the following expression for the normalization
#
# normalized_img = 255/( max(img) - min(img) )*( img -  min(img) )

resXY_norm2 = 255/( np.max(resXY) - np.min(resXY) )*( resXY -  np.min(resXY) )

print("Max absolute difference between the normalizations: %g"%np.max(np.abs(resXY_norm - resXY_norm2)))
