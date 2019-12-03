#%% package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% def loadImages
def displayFeatures2d(feat, group=None):
    '''
    Fonction permettant de visualiser les descripteurs en 2D
    appels possibles:
        - displayFeatures2d(feat)
        - displayFeatures2d(feat,group) si on a les classes d'appartenance de
        chaque pixel
    '''
    
    # if
    if group is None:
        group = 'b'
    
    
    #transform to pandas dataframe
    df = pd.DataFrame(feat, columns=['red', 'green', 'blue'])  

    # Affichage
    axes = pd.plotting.scatter_matrix(df,figsize=[10,10],c=group, diagonal='hist', grid=True)
    # Sur certaines versions de pandas, il faut enleevr tools.
    #axes = pd.plotting.scatter_matrix(df,figsize=[10,10],c=group, diagonal='hist', grid=True)

    #    axes.grid()
    #    plt.title('Nuage de points et histogramme')
    
    

    return axes