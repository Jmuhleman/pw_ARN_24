# **Rapport sur la classification des phases de sommeil chez la souris à l'aide des réseaux de perceptrons multicouches (MLP)**
# **Travail pratique 03**

**Cours : Apprentissage avec des réseaux neuronaux artificiels - ARN**<br>
**Professeur :** Andres Perez-Uribe <br>
**Assistants :** Shabnam Ataee, Simon Walther <br>
**Étudiants :** Julien Mühlemann, Lucas Lattion



'''text

dernière parti:

- ne pas chercher à améliorer le F1
- mentionner que le set de validation ne dispose que de 18 observations et donc qu'à chaque déviation
la validation est instable. (explication des graphes en dents de scie)

'''






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
- `EEG_mouse_data_test.csv` pour les prédictions de la compétition

Les étapes de prétraitement ont impliqué la normalisation des colonnes numériques à l'aide de `StandardScaler` pour garantir que les entrées du modèle ont une moyenne de zéro et un écart-type de un.
La normalisation est utile afin de donner un poids équivalent aux données d'entrées et ainsi améliorer la vitesse d'apprentissage lors du calcul des poids synaptiques.

### **Métriques de performance**
La performance du modèle a été évaluée à l'aide du score F1 et d'une matrice de confusion. Le score F1 moyen sur les folds a été calculé pour évaluer la performance globale. Nous avons un graphique de l'historique du loss durant l'entrainement et la validation sur chaque fold afin de bien visualiser l'évolution du modèle pendant son entrainement. Ainsi nous pouvons nous rendre compte si overfitting ou manque de généralisation il y a.


## <span id="2-first-experiment-awake--asleep">First experiment: awake / asleep</span>


La première expérience consiste à séparer les échantillons entre l'état éveillé et endormi.

### <span id="21-model-summary">Model summary</span>

![1_model_summary](1_model_summary.jpeg)

### <span id="22-résultat-et-performances">Résultat et performances</span>

![1_confusion_matrix](1_confusion_matrix.jpeg)

![1_training_validation_loss](1_training_validation_loss.png)

### <span id="23-analyse-des-résultats">Analyse des résultats</span>
Il semble assez facile d’obtenir un bon score avec uniquement 2 classes.
Notre F1-Score atteint un score de 90% sur le meilleur fold et de 80% en moyenne. 

- Nous pouvons en outre remarque que l'apprentissage se faire très vite, la propagation des erreurs nous mène très rapidement vers un loss de ~0.08.

- A epoch = 40, nous pouvons noter que le réseau est 'surpris' par le testset. Celui-ci retourne une moins bonne performance. Ceci pourrait survenir éventuellement à cause d'outlier qui ne se verraient pas bien classifiés.

- Vers la fin de l'entrainement epoch = 80 nous pouvons noter une divergence des courbes de test et d'entrainement. Ceci provient d'un overfitting qui commence à apparaitre à ce moment là. Le réseau ne parvient plus à généraliser la classification.

## <span id="3-second-experiment-awake--n-rem--rem">Second experiment: awake / n-rem / rem</span>


Pour cette seconde expérience nous allons effectuer une classification sur les 3 classes: W: awake, n: N-rem, r: rem.

### <span id="31-model-summary">Model summary</span>

![3_model_summary](2_model_summary.png)

### <span id="32-résultat-et-performances">Résultat et performances</span>

![3_confusion_matrix](2_confusion_matrix.png)

![3_training_validation_loss](2_training_validation_loss.png)

### <span id="33-analyse-des-résultats">Analyse des résultats</span>

Nous sommes passé de 2 à 3 classes en gardant la même architecture à savoir: 
* 3 neurones en couche caché
* tanh comme fonction d'activation
* un learning rate de 0.01
* un momentum de 0.99
* une loss fonction MSE.

Nos résultats étaient significativement moins bon (6% de F1 moyenné entre les 3 folds).

Nous avons constaté un underfitting puisque le modèle ne permettait pas de faire des prédiction d'une performance raisonnable.

Par conséquent nous avons:

* ajouté une couche de dropout 0.3 à 0.3
* ajusté le nombre de neurones cachés à 40
* changé les fonctions d'activations en relu pour la couche cachée et également les neurones de réponse
* le learning rate diminue à 0.0001 

Avec ces changements nous sommes parvenus à un F1 de ~58%

Nous avons choisi de diminuer le learning rate car l'entrainement ne demande pas une grande quantité de temps. En outre un minimum local n'a peut-être pas été trouvé durant l'entrainement précédent.
L'augmenter n'aurait fait certainement que rendre instable la mise à jour des poids.

Nous avons choisi d'augmenter la quantité de neurones dans la couche caché car le réseau précédent était under-fitté.

## <span id="4-competition-awake--n-rem--rem">Competition: awake / n-rem / rem</span>


Dans l'expérience #3 nous allons affiner le réseau établi au point précédent afin d'obtenir les meilleures performances possibles.

### <span id="41-model-summary">Model summary</span>

![3_model_summary](3_model_summary.png)

### <span id="42-résultat-et-performances">Résultat et performances</span>

![3_confusion_matrix](3_confusion_matrix.png)

![3_training_validation_loss](3_training_validation_loss.png)

### <span id="43-analyse-des-résultats">Analyse des résultats</span>

Les résultats sont bien meilleures pour cette expérience et monte jusqu'à 90% de F1-score.
Afin d'obtenir ces résultats nous avons apporté plusieurs optimisations:

#### Sélection des données :

Afin d'améliorer les résultats pour la compétition, nous avons décidé d'essayer de faire une meilleure sélection des features que l'on utilise.

Pour effectuer cela, nous avons utilisé la PCA. Celle-ci va nous retourner les components dont la variance est la plus grande, en espérant que la variance permet dans notre cas d'expliquer la classification.

Ceci nous permettra de réduire la dimensionnalité de notre dataset.
Après plusieurs essais nous avons constaté que 10 components était le paramètre qui nous offrait la meilleure performance.

Cette fois, nous nous sommes servis des 2 datasets fournis par notre professeur afin de faire un entrainement plus robuste sur environ 40'000 tuples.


#### Encodage des classes:

Nous avons choisi de procéder à un encodage one-hot afin d'avoir un neurone de sortie par classe. Étant donnée que nous n'avons pas notion de rang entre les 3 classes nous pensons que cela permettra au réseau de mieux classifier les états de nos chères souris clonées.

#### Architecture des couches :

Pour ce réseau nous avons décidé de monter le nombre de couches à 5. Si nous montons le nombre de neurones alors nous risquons l'overfitting.

En outre nous avons décidé de placer des couches de Dropout (2 exactement) ce qui désactiverait de manière aléatoire certains neurones lors de l'entrainement. La fin du réseau est paramétré pour le one-hot avec 1 neurone par classe.

### Fonctions d'activations:

Nous avons décidé de rester sur relu pour notre mlp. Après plusieurs essais avec tanh et sigmoid plutôt catastrophiques, nous sommes revenu sur relu. En effet, nous risquons le problème du 'vanishing gradient' si nous utilisons tanh ou la sigmoid dans les couches cachés. Les valeurs des dérivées pouvant être  < 1, leur produit se verra tendre vers 0.

Enfin, nous nous sommes servis de softmax pour la couche de sortie. Nous pensons qu'avec le one-hot, des valeurs probabilistes sont adaptées.

#### Fonction de loss :

Pour notre première tentative, nous avons mis en œuvre une perte de "mse" (Mean Squared Error), adaptée à un problème de régression. En revanche, dans notre deuxième itération, nous avons opté pour une perte de "categorical_crossentropy", mieux adaptée à un problème de classification multi-classe.

#### Optimiseur :

Notre premier modèle utilisait l'optimiseur SGD (Stochastic Gradient Descent) avec un taux d'apprentissage de 0.0001 et un momentum de 0.99. En revanche, dans notre deuxième version, nous avons fait le choix de l'optimiseur Adam avec ses paramètres par défaut, afin de bénéficier de ses performances et de sa stabilité.

#### Regularisation :

Pour renforcer la robustesse de notre modèle, nous avons introduit une régularisation L2 dans notre deuxième approche, avec un paramètre de régularisation de 0.01 appliqué à la couche Dense.


## <span id="5-conclusion">Conclusion</span>
En conclusion, ce rapport détaille notre expérience de classification des phases de sommeil chez la souris en utilisant des réseaux de perceptrons multicouches (MLP). Nous avons exploré deux expériences principales, d'abord en distinguant les phases d'éveil et de sommeil, puis en identifiant les phases d'éveil, de sommeil paradoxal (REM) et de sommeil profond (N-REM).

Nous avons constaté que la classification entre deux classes (éveil / sommeil) est relativement simple, avec des scores F1 atteignant jusqu'à 90%. Cependant, lorsque nous avons ajouté une troisième classe pour distinguer les différentes phases de sommeil, les performances ont diminué, nécessitant des ajustements tels que l'optimisation des hyperparamètres, l'utilisation de différentes fonctions de perte et l'introduction de techniques de régularisation et de dropout pour améliorer la généralisation du modèle.

Notre dernier modèle a montré des performances significativement améliorées, avec des scores F1 allant jusqu'à 90%, grâce à des améliorations telles que la sélection de données améliorée(PCA), l'utilisation de fonctions d'activations différentes et une fonctions de loss adaptée.

L'introduction de l'optimiseur Adam nous a permit d'atteindre d'affiner encore le résultat. Ce rapport reflète notre parcours pour affiner notre modèle de classification des phases de sommeil, en mettant en évidence les défis rencontrés et les solutions mises en œuvre pour obtenir des performances optimales.





