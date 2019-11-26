
#%% package
import numpy as np
import matplotlib.pyplot as plt

from loadImages import loadImages
from selectFeatureVectors import selectFeatureVectors
from displayFeatures2d import displayFeatures2d
from displayFeatures3d import displayFeatures3d


#%% def preprocessing
def preprocessing():
    # ------------------------------------------------
    # YOUR CODE HERE
    img73, img87 = loadImages()
    featLearn = 0

    # Redimensionnement
    img73 = img73[120:820, 180:800, :]
    img87 = img87[120:820, 180:800, :]

    # Affichage des images
    plt.figure(1);
    plt.imshow(img73);
    plt.show()

    plt.figure(2);
    plt.imshow(img87);
    plt.show()

    # Echantillonnage
    featLearn, nbPix, nbFeat = selectFeatureVectors(img73, 500)
    print('nb pix:', nbPix, '\r')
    print('nb feat:', nbFeat, '\r')
    # print('featLearn:', featLearn, '\r')

    # Affichage 2D et 3D
    displayFeatures2d(featLearn)
    displayFeatures3d(featLearn)

    # ------------------------------------------------
    # ------------------------------------------------
    #%% sortie
    return featLearn,img73,img87
    