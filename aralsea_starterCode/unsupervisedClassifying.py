# %% package
import numpy as np
from sklearn.cluster import KMeans


# %% def unsupervisedClassifying
def unsupervisedClassifying(model, feat):
    ''' classement/prédiction à partir d'un modèle de classement non supervisé
    feat est la matrice du jeu de données à classer
    label est la classe prédite
    '''

    # ------------------------------------------------
    label = model.predict(feat)
    # ------------------------------------------------
    return label
