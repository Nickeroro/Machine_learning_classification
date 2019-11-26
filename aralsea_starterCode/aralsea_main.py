# %% Machine Learning Class - Exercise Aral Sea Surface Estimation

# package
import numpy as np
import matplotlib.pyplot as plt

from displayFeatures2d import displayFeatures2d
from displayFeatures3d import displayFeatures3d
from displayImageLabel import displayImageLabel
from unsupervisedClassifying import unsupervisedClassifying
from unsupervisedTraining import unsupervisedTraining

plt.ioff()  # to see figure avant input

# ------------------------------------------------
# YOUR CODE HERE
from preprocessing import preprocessing

# ------------------------------------------------
# ------------------------------------------------
# %% Examen des données, prétraitements et extraction des descripteurs
# Chargement des données et prétraitements
featLearn, img73, img87 = preprocessing()

# %% Apprentissage / Learning / Training

# Apprentissage de la fonction de classement
# ------------------------------------------------
model = unsupervisedTraining(featLearn, 'kmeans')
# ------------------------------------------------

# prediction des labels sur la base d'apprentissage
# ------------------------------------------------
labels = model.labels_
# ------------------------------------------------

# Visualisation des resultats
# ------------------------------------------------
displayFeatures2d(featLearn, labels)
displayFeatures3d(featLearn, labels)
# ------------------------------------------------


# %% Classement et estimation de la diminution de surface
# Classifying / Predicting / Testing

# mise en forme de l'image de 1973 et 1987 en matrice Num Pixels / Val Pixels
# ------------------------------------------------
n_img73 = np.zeros([img73.shape[0] * img73.shape[1], 3])
for i in range(0, 2):
    n_img73[:, i] = img73[:, :, i].flatten()
# print(n_img73)

n_img87 = np.zeros([img87.shape[0] * img87.shape[1], 3])
for i in range(0, 2):
    n_img87[:, i] = img87[:, :, i].flatten()
# print(n_img73)
# ------------------------------------------------

# Classement des deux jeux de données et visualisation des résultats en image
# ------------------------------------------------
label_img73 = unsupervisedClassifying(model, n_img73)
label_img87 = unsupervisedClassifying(model, n_img87)

displayImageLabel(label_img73, img73)
displayImageLabel(label_img87, img87)
# ------------------------------------------------

# %% Estimation de la surface perdue
answer = input('Numero de la classe de la mer ? ')
cl_mer = int(answer)
# ------------------------------------------------
surface_73 = 0
for i in label_img73:
    if i == cl_mer:
        surface_73 += 1
print('surface en pixels de la classe: ', surface_73)

surface_87 = 0
for i in label_img87:
    if i == cl_mer:
        surface_87 += 1
print('surface en pixels de la classe: ', surface_87)

evo_surface = (1-surface_87/surface_73)*100
print('evolution de la surface: ', evo_surface, "%")
# ------------------------------------------------
