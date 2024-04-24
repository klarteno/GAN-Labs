# plot images saved in folder in a stack of images 
import cv2 as cv2
import matplotlib
import matplotlib.pyplot as plt



# try this if images display colour is not normal

def  display_images(resulting_images):
    ''' 
    from IPython.core.display import Image, display
    display(Image(file_path, width=700, unconfined=True))
    '''

    resulting_image = cv2.imread(resulting_images)
    resulting_image = cv2.cvtColor(resulting_image, cv2.COLOR_BGR2RGB)
    # Determine the figures size in inches to fit your image
    height, width, depth = resulting_image.shape
    dpi = matplotlib.rcParams['figure.dpi']

    figsize = width / float(dpi), height / float(dpi)

    print('height, width, depth: ',height, width, depth)

    plt.figure(figsize=figsize)
    plt.imshow(resulting_image)
    plt.show()

