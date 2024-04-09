# p6_classer_des_images
# CNN (réseaux de neurones convolutionnels), transfer learning

Le but de ce projet est de d'aider une association (fictive) de protection des animaux,
en mettant à leur disposition un algorithme capable de classer les images en fonction de la race du chien présent sur l'image.
Cet algorithme doit bien entendu être aussi fiable que possible, rapide et simple d'utilisation.


Etapes principales :

    Les images utilisées pour entrainer nos modèles proviennent du Stanford Dogs Dataset.
Ce datatset contient plus de 20 000 images, regroupées par races de chien.
    Point important pour nous : il n'est pas inclus dans le dataset "imagenet", 
qui a servi a entrainer les modeles que nous utiliserons en partie 3.


1) Prétraitement classique

    Le premier notebook contient l'exploration et le nettoyage des données, ainsi que plusieurs étapes de prétraitements 
spécifiques aux images : redimensionnement, ajustement du contraste et de l'exposition, adoucissement du bruit grâce à différents filtres
(gaussien, bilinéaire, non-local).
    Notons qu'aujourd'hui ces étapes sont directement intégrées dans les premières couches des réseaux neuronaux, ce qui présente plusieurs avantages
(simplicité, compatibilité, efficacité du pipeline).


2) Entrainement d'un modèle perso

    Dans le notebook 2_1, nous utilisons keras avec tensorflow en backend pour créer un premier modèle séquentiel très simple.
Il s'agit d'une architecture inspirée de LeNet 5, comprenant 2 couches de convolution + maxpooling, 1 flatten et 2 denses fully-connected,
pour un total de moins d'un million de paramètres.
    Ce modèle nous permet d'abord, dans une certaine mesure, d'analyser les prétraitements effectués.
Pour cela l'ui mlflow est très pratique.
    Nous pouvons ensuite commencer l'optimisation des hyperparamètres du modèle, avec keras tuner ou optuna par exemple :
nb et taille des filtres (couches conv), nombre de neurones pour la première couche f-c, fonction d'activation, optimizer.
    Pour pouvoir entrainer en parallèle plusieurs modèles (pour 3, 4, 5 classes), une copie de ce notebook, légèrement modifiée,
(imports, adresses locales remplacées par les adresses sur le drive) est compatible avec Colab.

    La recherche d'amélioration des modèles persos se poursuit dans le notebook 2_2, 
d'abord en ajoutant au modèles des couches de data augmentation,
puis en continuant l'optimisation des hyperparamètres (nombre d'epochs, épaisseur des couches de convolution, nombre de blocs conv,
dropout, batch size).

    En quelques jours il est possible d'entrainer des modèles aux résultats significatifs (accuracy entre 0.7 et 0.8), mais seulement
sur un nombre très limité de classes. Pour obtenir rapidement un résultat bien meilleur, nous allons passer au transfer learning
sur des modèles préentrainés. 


3) Tansfer learning

    Ce notebook permet de fine-tuner sur nos données plusieurs modèles :
    
- un modèle classique, VGG_16
- les 4 modèles les + performants parmi tous ceux disponibles sur keras.application (évaluéation sur ImageNet validation dataset) :
'EfficientNetV2L', 'NASNetLarge', 'EfficientNetV2S', 'InceptionResNetV2'
- les 2 meilleurs parmi les modèles + légers : 'NASNetMobile' et 'MobileNetV2'

    Une fois le meilleur modèle choisi, il ne reste plus qu'à le déployer !


4) Applis

    Deux applications réalisées avec streamlit :
    
    La première, app_test.py, permet de tester en local les prédictions du modèle choisi, 
en tirant au hasard une image (non utilisée pour l'entrainement), 
puis en affichant sa classe réelle et les 2 premières classes prédites par le modèle (+ probas associées).

    La seconde, app_asso.py, est une version + conforme au besoin réel de notre association imaginaire :
Le modèle renvoit une prédiction pour toute image uploadée (drag and drop). 
C'est cette version qui est déployée, avec un tracking lfs pour notre modèle, à l'adresse :

https://image-classification-dog-breeds.streamlit.app/ 




