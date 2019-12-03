#%% Machine Learning Class - Exercise Aral Sea Surface Estimation

# package
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# ------------------------------------------------
# YOUR CODE HERE
from preprocessing import preprocessing
from unsupervisedTraining import unsupervisedTraining
from displayFeatures2d import displayFeatures2d
from displayFeatures3d import displayFeatures3d
from selectFeatureVectors import selectFeatureVectors
from unsupervisedClassifying import unsupervisedClassifying
from displayImageLabel import displayImageLabel

# ------------------------------------------------
# ------------------------------------------------

###------------ METHODE KMEANS ------------###

#%% Examen des données, prétraitements et extraction des descripteurs

# Chargement des données et prétraitements
featLearn,img73,img87 = preprocessing()

#%% Apprentissage / Learning / Training
# Apprentissage de la fonction de classement
# ------------------------------------------------
# YOUR CODE HERE
model = unsupervisedTraining(featLearn, method='gmm')
# ------------------------------------------------
# ------------------------------------------------
# prediction des labels sur la base d'apprentissage
# ------------------------------------------------
# YOUR CODE HERE
pred = model.predict(featLearn)
# ------------------------------------------------
# ------------------------------------------------
# Visualisation des resultats
displayFeatures2d(featLearn,pred)
displayFeatures3d(featLearn,pred)
# ------------------------------------------------

#%% Classement et estimation de la diminution de surface
# Classifying / Predicting / Testing


# mise en forme de l'image de 1973 et 1987 en matrice Num Pixels / Val Pixels

# ------------------------------------------------
# YOUR CODE HERE
feat_image_73, nbre_pixels_image, nbr_couche_image = selectFeatureVectors(img73,1)
feat_image_87, nbre_pixels_image, nbr_couche_image = selectFeatureVectors(img87,1)
# ------------------------------------------------
# ------------------------------------------------

# Classement des deux jeux de données et visualisation des résultats en image

# ------------------------------------------------
# YOUR CODE HERE
label_img73 = unsupervisedClassifying(model,feat_image_73)
#displayFeatures2d(feat_image_73,label_img73)
#displayFeatures3d(feat_image_73,label_img73)

label_img87 = unsupervisedClassifying(model,feat_image_87)
displayImageLabel(label_img73,img73)
displayImageLabel(label_img87,img87)

# ------------------------------------------------
# ------------------------------------------------

#%% Estimation de la surface perdue
answer = input('Numero de la classe de la mer ? ')
cl_mer = int(answer)

# ------------------------------------------------
# YOUR CODE HERE
surface_73 = 0
for i in label_img73:
    if i == cl_mer:
        surface_73 += 1
print('surface en pixels de la classe image73: ', surface_73)

surface_87 = 0
for i in label_img87:
    if i == cl_mer:
        surface_87 += 1
print('surface en pixels de la classe image87: ', surface_87)

evo_surface = (1-surface_87/surface_73)*100
print('evolution de la surface kmeans: ', evo_surface, "%")
# ------------------------------------------------

# ------------------------------------------------
# ------------------------------------------------

###------------ GAUSSIAN MIXTURE ------------###

#%% Apprentissage / Learning / Training
# Apprentissage de la fonction de classement
# ------------------------------------------------
# YOUR CODE HERE
model = unsupervisedTraining(featLearn, method='gmm')
# ------------------------------------------------
# ------------------------------------------------
# prediction des labels sur la base d'apprentissage
# ------------------------------------------------
# YOUR CODE HERE
pred = model.predict(featLearn)
# ------------------------------------------------
# ------------------------------------------------
# Visualisation des resultats
displayFeatures2d(featLearn,pred)
displayFeatures3d(featLearn,pred)
# ------------------------------------------------

#%% Classement et estimation de la diminution de surface
# Classifying / Predicting / Testing


# mise en forme de l'image de 1973 et 1987 en matrice Num Pixels / Val Pixels

# ------------------------------------------------
# YOUR CODE HERE
feat_image_73, nbre_pixels_image, nbr_couche_image = selectFeatureVectors(img73,1)
feat_image_87, nbre_pixels_image, nbr_couche_image = selectFeatureVectors(img87,1)
# ------------------------------------------------
# ------------------------------------------------

# Classement des deux jeux de données et visualisation des résultats en image

# ------------------------------------------------
# YOUR CODE HERE
label_img73 = unsupervisedClassifying(model,feat_image_73)
#displayFeatures2d(feat_image_73,label_img73)
#displayFeatures3d(feat_image_73,label_img73)

label_img87 = unsupervisedClassifying(model,feat_image_87)
displayImageLabel(label_img73,img73)
displayImageLabel(label_img87,img87)
# ------------------------------------------------
# ------------------------------------------------
# ------------------------------------------------
# YOUR CODE HERE
answer = input('Numero de la classe de la mer ? ')
cl_mer = int(answer)
surface_73 = 0
for i in label_img73:
    if i == cl_mer:
        surface_73 += 1
print('surface en pixels de la classe image73: ', surface_73)

surface_87 = 0
for i in label_img87:
    if i == cl_mer:
        surface_87 += 1
print('surface en pixels de la classe image87: ', surface_87)

evo_surface = (1-surface_87/surface_73)*100
print('evolution de la surface méthode gaussian mixture: ', evo_surface, "%")
# ------------------------------------------------

# ------------------------------------------------
# ------------------------------------------------