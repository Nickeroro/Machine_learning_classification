#%% package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#%% def displayFeatures3d
def displayFeatures3d(feat, group=None):
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
    axes = plt.figure(4).gca(projection='3d')
    axes.scatter(df['red'], df['green'], df['blue'], c=group)
    axes.set_xlabel('red')
    axes.set_ylabel('green')
    axes.set_zlabel('blue')
    #    axes.grid()
    plt.title('Nuage de points et histogramme')
    plt.show()
    

    return axes


