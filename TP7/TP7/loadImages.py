#%% package
import numpy as np
import cv2
import matplotlib.pyplot as plt


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
    im73 = cv2.imread(im73_filename)
    im87 = cv2.imread(im87_filename)
    # si les lignes d'au-dessus ne fonctionnent pas
    # im73 = plt.imread(im73_filename)
    # im87 = plt.imread(im87_filename)




  #%% sortie

    return im73, im87