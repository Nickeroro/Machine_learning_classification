
#%% package
import numpy as np
import matplotlib.pyplot as plt

from loadImages import loadImages
from selectFeatureVectors import selectFeatureVectors


#%% def preprocessing
def preprocessing():
    
    # ------------------------------------------------
    # YOUR CODE HERE
    img73, img87 = loadImages()
    
    img73 = img73[65:850,:,:]
    img87 = img87[65:850,:,:]
    
    plt.figure()
    plt.subplot(121)
    plt.imshow(img73)
    plt.subplot(122)
    plt.imshow(img87)
    
    # ------------------------------------------------
    # ------------------------------------------------
    pas_echantillonage = 500
    featLearn, nbre_pixels, nbr_couche = selectFeatureVectors(img73,pas_echantillonage)
        
    #%% 
    return featLearn,img73,img87
    