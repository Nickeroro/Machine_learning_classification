#%% package
import numpy as np


#%% def selectFeatureVectors
def selectFeatureVectors(img,nbSubSample):
    '''Fonction permettant de récupérer l'image sous la forme d'un vecteur en
    sous échantillonnant  
    '''
    
    
    # Locales
    nbLin,nbCol,nbFeat = img.shape
    
    
    #%% Extraction des trois composantes de l'images
    nbPix = nbLin*nbCol
    
    vr = img[:,:,0].reshape((nbPix,))
    vg = img[:,:,1].reshape((nbPix,))
    vb = img[:,:,2].reshape((nbPix,))
        
    # Mise sous la forme d'un vecteur
    feat = np.ones((nbPix,nbFeat))
    feat[:,0] = vr
    feat[:,1] = vg
    feat[:,2] = vb
    
    
    #%% sous-echantillonnage
    
    # indice des échantillons à garder
    ivect = np.arange(0, nbPix, nbSubSample)
    
    # sélection
    feat = feat[ivect,:]
    
    # nombre de pixels gardés 
    nbPix = ivect.shape[0]
        
    
    #%% sortie    
    return feat,nbPix,nbFeat
