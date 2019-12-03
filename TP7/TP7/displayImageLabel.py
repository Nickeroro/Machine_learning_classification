# useful packages
import matplotlib.pyplot as plt


# def displayImageLabel
def displayImageLabel(label,img):
    '''
    fonction qui affiche l'image résultat de classification
    '''
    plt.figure()
    # locales
    nbLig,nbCol,nbComp = img.shape
    
    # mise en forme
    imgLabel = label.reshape((nbLig,nbCol))
    
    # affichage
    imgplot = plt.imshow(imgLabel)
    
    # options d'affichage
    plt.colorbar()
    
    return imgLabel, imgplot

