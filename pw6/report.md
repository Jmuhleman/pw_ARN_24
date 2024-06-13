<img src="https://storage.googleapis.com/visual-identity/logo/2020-slim.svg" style="height:80px;"><br><br><br>


# Object recognition in the wild using Convolution Neural Networks

**Cours : Apprentissage avec des réseaux neuronaux artificiels - ARN**<br>
**Professeur :** Andres Perez-Uribe <br>
**Assistants :** Shabnam Ataee, Simon Walther <br>
**Étudiants :** Julien Mühlemann, Lucas Lattion


## Table des matières

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [Object recognition in the wild using Convolution Neural Networks](#object-recognition-in-the-wild-using-convolution-neural-networks)
  - [Table des matières](#table-des-matières)
  - [1. Introduction](#1-introduction)
  - [2. Problématique](#2-problématique)
  - [3. Préparation des données](#3-préparation-des-données)
  - [4. Création du modèle](#4-création-du-modèle)
    - [4.1 Architecture](#41-architecture)
    - [4.2 Transfer learning](#42-transfer-learning)
  - [5. Résultats](#5-résultats)
  - [6. Conclusion](#6-conclusion)

<!-- /code_chunk_output -->

<!-- pagebreak -->

## 1. Introduction

Dans le cadre de ce laboratoire, nous allons suivre toutes les étapes nécessaires pour créer une application de classification d’objets. Pour ce faire, nous avons réfléchi à ce qu'il serait utile de classifier et avons décidé de nous concentrer sur les différents billets de banque suisses.

En plus de nous offrir une première expérience du transfert de l’apprentissage (transfer learning), une telle application pourrait permettre la mise en place de systèmes de reconnaissance automatique pour des raisons de sécurité ou d’analyse financière, et même aider les amateurs de numismatique à identifier les différents billets en circulation en Suisse.

En nous basant sur un dataset préexistant, retravaillé pour correspondre à nos besoins, et sur des photos prises par nos soins, nous allons appliquer le transfert de l’apprentissage à l’aide de MobileNetV2 sur les poids d'ImageNet dont les couches sont figées. Nous procéderons ensuite au peaufinage du modèle afin d’obtenir les meilleurs résultats possibles avec notre dataset.

## 2. Problématique
Nous étions initialement partis dans l’idée de différencier les billets de banque internationaux, avant de nous rendre compte de la complexité de la collecte de telles données. En effet, outre le grand nombre de devises, plusieurs d’entre elles présentent des similitudes frappantes, rendant une telle classification très difficile pour une première expérience du transfert de l’apprentissage.

Nous avons donc choisi de nous concentrer sur les différents types de billets de banque suisses. Chaque billet possède des caractéristiques reconnaissables, suivant les éléments graphiques et les couleurs spécifiques à chaque dénomination. Une illustration des billets sélectionnés se trouve dans la figure 1.

Après plusieurs recherches sur internet, nous avons trouvé des datasets existants pour la classification des billets de banque et nous en avons utilisé un provenant de Kaggle, offrant une séparation des images par type de billet. Ce dataset, basé sur le dataset de Swiss Banknotes Dataset, contient plus de 8’000 images réparties en plusieurs classes, dont celles de notre sélection. Le nombre d’images de ce dataset pour nos classes varie entre 250 et 1’400 images. Nous avons donc conservé uniquement une partie d’elles afin d’équilibrer et de réduire le dataset que nous allons utiliser.

## 3. Préparation des données
Lors de la sélection des photos pour notre ensemble de données, nous avons conservé les photos ayant peu d’éléments perturbateurs en arrière-plan, comme d’autres objets ou des inscriptions.

Nous avons également essayé de mélanger différentes séries de billets, pour que le modèle ait une meilleure chance de reconnaître les caractéristiques spécifiques à chaque type de billet. Nous avons sélectionné aléatoirement 20 % des images pour notre ensemble de test, et les 80 % restants pour l’entraînement. Les images d’entraînement auront un traitement additionnel pour ajouter des variations, que notre ensemble de validation n’aura pas.

Pour introduire des variations dans les données, nous avons appliqué des effets de miroir, de contraste, de zoom et de rotations à une partie des images de test. Nos images n’étant pas nécessairement uniformes, ces modifications apportent toutefois une possibilité au modèle de se concentrer sur les éléments des billets qui caractérisent leur type, au lieu de se concentrer uniquement sur les formes et les couleurs. Toutes les images sont redimensionnées au format 224x224 pixels pour avoir une taille uniforme en entrée du modèle.

## 4. Création du modèle
### 4.1 Architecture
Notre modèle utilise MobileNetV2 avec les poids d’ImageNet, sur lequel nous allons rajouter une couche dense de 128 neurones qui sera la seule couche paramétrable, les couches de MobileNetV2 restant figées. Étant donné l’ensemble de caractéristiques importantes dans les photos du dataset, nous avons fait le choix d’ajouter une régularisation L1 et L2, respectivement lasso regression et ridge regression, sur notre unique couche dense en sortie de MobileNetV2. Cet effet additionnel sur la fonction de coût aide le modèle à éliminer certaines caractéristiques inutiles à la classification, tout en limitant le risque d’overfitting.

Nous avons effectué quelques essais avec notre ensemble de données en introduisant certaines variations dans les hyper-paramètres pour trouver un modèle optimal. Le modèle suivant a été sélectionné :

Nombre d’epochs : 8.
Optimizer : RMSprop conservé du modèle fourni dans la première partie.
Learning rate : 0.001 (par défaut pour RMSprop).
Couches denses : 1 couche avec 128 neurones et une régularisation L1 et L2.
Fonction d’activation : ReLU.
La performance de l’entraînement est illustrée dans la figure 4. Nous avons noté un certain overfitting dans les dernières epochs, que nous avons cherché à réduire en ajoutant de la régularisation L1 et L2, ainsi qu’en introduisant un effet dropout après notre couche dense en sortie du modèle. Cela a bien aidé la précision, bien que cela ait introduit une variation plus élevée dans les résultats de la fonction de coût. Les expériences suivantes sur ce modèle restant quand même satisfaisantes, et compte tenu de la difficulté de notre choix initial, nous avons décidé que ce modèle serait suffisant.

Ce modèle possède un total de 2,422,468 paramètres, dont seulement 164,484 sont entraînables, le reste étant les poids du modèle ImageNet.

### 4.2 Transfer learning
Pour effectuer le transfert de l’apprentissage, nous avons utilisé MobileNetV2 avec les poids d’ImageNet. Étant donné la petite taille de notre dataset, tous les paramètres d’ImageNet ont été figés, ne laissant que la couche dense comme étant paramétrable. Notre dataset n’étant toutefois pas totalement similaire aux objets utilisés pour ImageNet, nous aurions probablement pu laisser une partie des paramètres du modèle entraînables, mais nous avons décidé de nous concentrer principalement sur la modification des hyper-paramètres.

Le transfert de l’apprentissage dans notre cas est très utile. En effet, c’est un moyen efficace d’obtenir un modèle performant capable de différencier certaines caractéristiques d’une photo et d’analyser de manière plus précise un petit ensemble d’images. Un modèle convolutif classique entraîné sur notre dataset aurait donné des résultats de très mauvaise qualité, avec un overfitting inévitable.

## 5. Résultats
La performance de l’entraînement est donnée dans la figure 4. La matrice de confusion du système ainsi que les résultats de validation sont dans la figure 5.

Comme nous pouvons l’observer, le modèle a plus de difficultés avec les prédictions de certaines classes de billets. Cela semble logique car certaines dénominations peuvent avoir des similarités graphiques. Toutefois, nous sommes plutôt satisfaits de la performance de notre modèle, qui affiche un f-score relativement bon pour plusieurs classes de billets. Cela nous rassure quant à la capacité du modèle à séparer les caractéristiques spécifiques aux différents types de billets.

On note une fonction de coût relativement élevée, probablement due à la régularisation L1 et L2 appliquée dans la couche dense du modèle. Le facteur de différence entre le test et l’entraînement est cependant relativement correct, avec un coût de validation 1.2 fois plus élevé, ce qui n’indique pas nécessairement un overfitting sévère.

En application réelle, le modèle affiche quelques incertitudes, mais une fois la position de la caméra stabilisée, les pourcentages de classification sont relativement stables. Les essais réels ont montré quelques difficultés avec certains types de billets, notamment ceux de petites dénominations souvent confondus avec des billets de dénominations proches.


## 6. Conclusion
Ce laboratoire a été très intéressant, bien que notre choix de classes ait été probablement trop complexe. Un ensemble de classes plus facile et plus proche des objets reconnus par ImageNet aurait probablement causé moins de problèmes dans la classification. Nous avons rencontré plusieurs difficultés lors de la mise en place du dataset pour obtenir quelque chose d’intéressant pour le modèle, et cela s’est répercuté sur les résultats finaux. Parmi ces difficultés, on retrouve évidemment le choix de nos classes, dont la diversité intra-classe et les similarités inter-classes rendent la classification plus compliquée que si nous avions choisi des objets plus ordinaires.

Nous sommes toutefois grandement satisfaits des résultats obtenus et pensons que l’expérience nous aura formés sur l’utilisation du transfert de l’apprentissage ainsi que sur l’importance de la sélection initiale des classes et du dataset.

Nous pensons que les résultats valident notre problématique initiale, et, considérant que ImageNet est entraîné sur des objets ordinaires du quotidien, la classification finale est satisfaisante pour une première expérience sur le transfert de l’apprentissage et les réseaux convolutifs complexes.

La classification telle qu’effectuée ici présente des limites. En effet, les types de billets autres que ceux sélectionnés ne pourront pas être classifiés correctement. L’utilisation de ce modèle reste donc restreinte dans l’état actuel. Un travail futur pourrait partir des observations effectuées en incorporant un set de données plus grand et avec un nombre plus élevé de types de billets, en considérant les séries de billets plus récentes, que nous n’avons pas pu inclure dans notre dataset. Il pourrait également être utile de fournir une catégorie supplémentaire « autre » pour les objets qui ne sont pas des billets.

En résumé, nous sommes extrêmement satisfaits de nous être engagés dans le développement d’un cycle complet d’une application de classification d’objets. La navigation à travers la procédure complète jusqu’aux évaluations finales sur un appareil mobile nous a éclair
