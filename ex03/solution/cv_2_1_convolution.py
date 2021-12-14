import cv2 
import numpy as np

def convolve2D(image, kernel, padding=0, strides=1):
    """ 
        taken from
        https://medium.com/analytics-vidhya/2d-convolution-using-python-numpy-43442ff5f381
        and fixed the indexing for the output image
    """
    # Do convolution instead of Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    xKernLeft = xKernShape // 2
    # since slicing end is exclusive, uneven kernel shapes would be too small
    xKernRight = int(np.around(xKernShape / 2.))
    yKernShape = kernel.shape[1]
    yKernUp = yKernShape // 2
    yKernDown = int(np.around(yKernShape / 2.))
    xImgShape = image.shape[1]
    yImgShape = image.shape[0]

    # Shape of Output Convolution
    # START TODO ###################
    # xOutput =
    # yOutput = 
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    # END TODO ###################
    # output = np.zeros((xOutput, yOutput))
    output = np.zeros((yOutput, xOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        # START TODO ###################
        # imagePadded = 
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        # print(imagePadded)
        # END TODO ###################
    else:
        imagePadded = image

    # Indices for output image
    x_out = y_out = -1
    # Iterate through image
    for y in range(yKernUp, imagePadded.shape[0], strides):
        # print(y)
        # exit()
        # START TODO ###################
        # Exit Convolution before y is out of bounds
        if y > imagePadded.shape[0] - yKernDown:
            break
        # END TODO ###################
        
        # START TODO ###################
        # iterate over columns and perform convolution
        # position the center of the kernel at x,y
        # and save the sum of the elementwise multiplication
        # to the corresponding pixel in the output image
        y_out += 1
        for x in range(xKernLeft, imagePadded.shape[1], strides):
            # Go to next row once kernel is out of bounds
            if x > imagePadded.shape[1] - xKernRight:
                break

            x_out += 1
            x_out_idx = x_out % xOutput
            y_out_idx = y_out % yOutput
            # output[x_out_idx, y_out_idx] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
            output[y_out_idx, x_out_idx] = (kernel * imagePadded[y-yKernUp:y+yKernDown, x-xKernLeft: x+xKernRight]).sum()
        # END TODO ###################
    return output


if __name__ == '__main__':
    # Grayscale Image
    image = cv2.imread('image.png',0)

    # Edge Detection Kernel
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    # Convolve and Save Output
    output = convolve2D(image, kernel, padding=2, strides=2)
    cv2.imwrite('2DConvolved.png', output)
