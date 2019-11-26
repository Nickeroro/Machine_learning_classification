#%% package
import numpy as np
from matplotlib.pyplot import imread


#%% def loadImages
def loadImages():
    ''' fonction de lecture des images 
    de la mer d'Aral 1973 et 1987
    '''

    # definition du chemin aux images
    path='';
    im73_filename = 'Aral1973_Clean.jpg';
    im87_filename = 'Aral1987_Clean.jpg';
    
    # lectures des images
    im73 = imread(im73_filename)
    im87 = imread(im87_filename)


  #%% sortie

    return im73, im87