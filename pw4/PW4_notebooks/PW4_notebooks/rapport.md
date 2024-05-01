# **Rapport sur le learning with Artificial Neural Networks**
# **Travail pratique 04**

**Cours : Apprentissage avec des réseaux neuronaux artificiels - ARN**<br>
**Professeur :** Andres Perez-Uribe <br>
**Assistants :** Shabnam Ataee, Simon Walther <br>
**Étudients :** Julien Mühlemann, Lucas Lattion

# Table des Matières

1. [Introduction](#1-introduction)
2. [First experiment: awake / asleep](#2-first-experiment-awake--asleep)
    - 2.1 [Model summary](#21-model-summary)
    - 2.2 [Résultat et performances](#22-résultat-et-performances)
    - 2.3 [Analyse des résultats](#23-analyse-des-résultats)
3. [Second experiment: awake / n-rem / rem](#3-second-experiment-awake--n-rem--rem)
    - 3.1 [Model summary](#31-model-summary)
    - 3.2 [Résultat et performances](#32-résultat-et-performances)
    - 3.3 [Analyse des résultats](#33-analyse-des-résultats)
4. [Competition: awake / n-rem / rem](#4-competition-awake--n-rem--rem)
    - 4.1 [Model summary](#41-model-summary)
    - 4.2 [Résultat et performances](#42-résultat-et-performances)
    - 4.3 [Analyse des résultats](#43-analyse-des-résultats)
5. [Conclusion](#5-conclusion)








1. What is the learning algorithm being used to optimize the weights of the neural 
networks? 
What are the parameters (arguments) being used by that algorithm? 
What loss function is being used ? 
Please, give the equation(s)
2. For each experiment excepted the last one (shallow network learning from raw data, 
shallow network learning from features and CNN):
1. Select a neural network topology and describe the inputs, indicate how many are 
they, and how many outputs?
2. Compute the number of weights of each model (e.g., how many weights between the 
input and the hidden layer, how many weights between each pair of layers, biases, 
etc..) and explain how do you get to the total number of weights.
3. Test at least three different meaningful cases (e.g., for the MLP exploiting raw data, 
test different models varying the number of hidden neurons, for the feature-based 
model, test pix_p_cell 4 and 7, and number of orientations or number of hidden 
neurons, for the CNN, try different number of neurons in the feed-forward part) 
describe the model and present the performance of the system (e.g., plot of the 
evolution of the error, final evaluation scores and confusion matrices). Comment the 
differences in results. Are there particular digits that are frequently confused?
3. The CNNs models are deeper (have more layers), do they have more weights than the 
shallow ones? explain with one example.
4. Train a CNN for the chest x-ray pneumonia recognition. In order to do so, complete the 
code to reproduce the architecture plotted in the notebook. Present the confusion matrix, 
accuracy and F1-score of the validation and test datasets and discuss your results.
 2


















# OLD RAPPORT

# **Rapport sur la classification des phases de sommeil chez la souris à l'aide des réseaux de perceptrons multicouches (MLP)**
# **Travail pratique 03**

**Cours : Apprentissage avec des réseaux neuronaux artificiels - ARN**<br>
**Professeur :** Andres Perez-Uribe <br>
**Assistants :** Shabnam Ataee, Simon Walther <br>
**Étudients :** Julien Mühlemann, Lucas Lattion

# Table des Matières

1. [Introduction](#1-introduction)
2. [First experiment: awake / asleep](#2-first-experiment-awake--asleep)
    - 2.1 [Model summary](#21-model-summary)
    - 2.2 [Résultat et performances](#22-résultat-et-performances)
    - 2.3 [Analyse des résultats](#23-analyse-des-résultats)
3. [Second experiment: awake / n-rem / rem](#3-second-experiment-awake--n-rem--rem)
    - 3.1 [Model summary](#31-model-summary)
    - 3.2 [Résultat et performances](#32-résultat-et-performances)
    - 3.3 [Analyse des résultats](#33-analyse-des-résultats)
4. [Competition: awake / n-rem / rem](#4-competition-awake--n-rem--rem)
    - 4.1 [Model summary](#41-model-summary)
    - 4.2 [Résultat et performances](#42-résultat-et-performances)
    - 4.3 [Analyse des résultats](#43-analyse-des-résultats)
5. [Conclusion](#5-conclusion)


## <span id="1-introduction">Introduction</span>
Dans cette séance pratique, nous avons utilisé des perceptrons multicouches (MLP) à l'aide de l'API Keras pour classifier les phases de sommeil des souris. Les données comprennent des informations provenant de plusieurs souris, qui sont des clones génétiques, garantissant une variance minimale des facteurs biologiques influençant les lectures EEG.


### Prétraitement des données
Nous avons traité des données EEG provenant de trois fichiers :
- `EEG_mouse_data_1.csv` et `EEG_mouse_data_2.csv` pour l'entraînement
- `EEG_mouse_data_test.csv` pour les tests

Les étapes de prétraitement ont impliqué la normalisation des colonnes numériques à l'aide de `StandardScaler` pour garantir que les entrées du modèle ont une moyenne de zéro et un écart-type de un.
La normalisation est utile afin de donner un poid équivalent aux données d'entrées et ainsi améliorer la vitesse d'apprentissage lors du calcul des poids synaptics.

### **Métriques de performance**
La performance du modèle a été évaluée à l'aide du score F1 et d'une matrice de confusion. Le score F1 moyen sur les plis a été calculé pour évaluer la performance globale, en plus des tracés de la moyenne et de l'écart-type des pertes d'entraînement et de validation sur les époques pour analyser la convergence du modèle et le risque de surajustement potentiel.


## <span id="2-first-experiment-awake--asleep">First experiment: awake / asleep</span>


La première expérence consiste à séparer les échantillons entre l'état éveillé et endormi.

### <span id="21-model-summary">Model summary</span>

![1_model_summary](1_model_summary.jpeg)

### <span id="22-résultat-et-performances">Résultat et performances</span>

![1_confusion_matrix](1_confusion_matrix.jpeg)

![1_training_validation_loss](1_training_validation_loss.jpeg)

### <span id="23-analyse-des-résultats">Analyse des résultats</span>
Il semble assez facile d'optenir un bon score avec uniquement 2 classes.
Notre F1-Score atteind un score de 90% sur le meilleur fold et de 80% en moyenne. 


## <span id="3-second-experiment-awake--n-rem--rem">Second experiment: awake / n-rem / rem</span>


Pour cette seconde expérience nous devons prendre les 3 classes éveillé, le sommeil paradoxal (REM) et le sommeil profond (N-REM) avec des étiquettes différentes. 

### <span id="31-model-summary">Model summary</span>

![3_model_summary](2_model_summary.png)

### <span id="32-résultat-et-performances">Résultat et performances</span>

![3_confusion_matrix](2_confusion_matrix.png)

![3_training_validation_loss](2_training_validation_loss.png)

### <span id="33-analyse-des-résultats">Analyse des résultats</span>

En passant de 2 à 3 classes nos résultats sont soudainement très médiocre.

En augmentant le nombre de neurone, le résultat s'est amélioré, mais la courbe d'entrainement et de validation diverge montrant un surapprentissage.

Pour corriger cela nous avons baisser le learning rate et le momentum, ainsi la validation ne diverge plus et les résultats s'améliore.


## <span id="4-competition-awake--n-rem--rem">Competition: awake / n-rem / rem</span>


La troisième expérience reprends l'expérience numéro 2 et identifie également les 3 classes éveillé, le sommeil paradoxal (REM) et le sommeil profond (N-REM).
Nous expliquons les améliorations que nous avons apporté dans la section "Analyse des résultats".

### <span id="41-model-summary">Model summary</span>

![3_model_summary](3_model_summary.jpeg)

### <span id="42-résultat-et-performances">Résultat et performances</span>

![3_confusion_matrix](3_confusion_matrix.jpeg)

![3_training_validation_loss](3_training_validation_loss.jpeg)

### <span id="43-analyse-des-résultats">Analyse des résultats</span>

Les résultats sont bien meilleures pour cette expérience et monte jusqu'à 90% de F1-score.
Afin d'optenir ces résultats nous avons apporté plusieurs optimisations:

#### Sélection des données :
Afin d'amélioré les résultats pour la compétitions, la première étape est d'améliorer la qualité des données que l'on donne à notre modèle. Pour effectuer cela, nous avons utilisé la fonction PCA (Principal Component Analysis) de sklearn qui sélectionne les fréquences avec le plus de variation.

#### Activation des couches :

Dans notre première version, nous avons choisi l'activation "relu" pour nos couches cachées, en maintenant également "relu" pour la couche de sortie. Cependant, pour notre deuxième modèle, nous avons décidé d'utiliser "relu" pour les couches de convolution et "softmax" pour la couche de sortie, dans le but de générer des probabilités pour chaque classe.

#### Fonction de perte :

Pour notre première tentative, nous avons mis en œuvre une perte de "mse" (Mean Squared Error), adaptée à un problème de régression. En revanche, dans notre deuxième itération, nous avons opté pour une perte de "categorical_crossentropy", mieux adaptée à un problème de classification multi-classe.

#### Optimiseur :

Notre premier modèle utilisait l'optimiseur SGD (Stochastic Gradient Descent) avec un taux d'apprentissage de 0.0001 et un momentum de 0.99. En revanche, dans notre deuxième version, nous avons fait le choix de l'optimiseur Adam avec ses paramètres par défaut, afin de bénéficier de ses performances et de sa stabilité.

#### Regularisation :

Pour renforcer la robustesse de notre modèle, nous avons introduit une régularisation L2 dans notre deuxième approche, avec un paramètre de régularisation de 0.01 appliqué à la couche Dense.

#### Dropout :

Dans notre deuxième modèle, nous avons incorporé une couche de dropout avec un taux de 0.2, afin de réduire le surapprentissage potentiel et d'améliorer la généralisation du modèle.


## <span id="5-conclusion">Conclusion</span>
En conclusion, ce rapport détaille notre expérience de classification des phases de sommeil chez la souris en utilisant des réseaux de perceptrons multicouches (MLP). Nous avons exploré deux expériences principales, d'abord en distinguant les phases d'éveil et de sommeil, puis en identifiant les phases d'éveil, de sommeil paradoxal (REM) et de sommeil profond (N-REM). Nous avons constaté que la classification entre deux classes (éveil / sommeil) est relativement simple, avec des scores F1 atteignant jusqu'à 90%. Cependant, lorsque nous avons ajouté une troisième classe pour distinguer les différentes phases de sommeil, les performances ont diminué, nécessitant des ajustements tels que l'optimisation des hyperparamètres, l'utilisation de différentes fonctions de perte et l'introduction de techniques de régularisation et de dropout pour améliorer la généralisation du modèle. Notre dernier modèle a montré des performances significativement améliorées, avec des scores F1 allant jusqu'à 90%, grâce à des améliorations telles que la sélection de données améliorée, l'utilisation d'activations et de fonctions de perte adaptées, ainsi que l'optimisation des hyperparamètres et l'introduction de régularisation et de dropout. Ce rapport reflète notre parcours pour affiner notre modèle de classification des phases de sommeil, en mettant en évidence les défis rencontrés et les solutions mises en œuvre pour obtenir des performances optimales.

