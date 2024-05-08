# **Deep neural networks**
# **Practical work #04**

**Class : Artificial Neural Network ARN**<br>
**Professor :** Andres Perez-Uribe <br>
**Assistants :** Shabnam Ataee, Simon Walther <br>
**Students :** Julien Mühlemann, Lucas Lattion



dernière partie:
```
- ne pas chercher à améliorer le F1
- mentionner que le set de validation ne dispose que de 18 observations et donc qu'à chaque déviation
la validation est instable. (explication des graphes en dents de scie)
```
## Introduction




## Overall description
> What is the learning algorithm being used to optimize the weights of the neural networks?

The learning algorithm used in the various experiences are :

*
*
*
*

> What are the parameters (arguments) being used by that algorithm?

The following parameters had been used for each algorithms :
*
*
*
*


> What loss function is being used ?

The loss functions are the following :
*
*
*
*

> Please, give the equation(s)

*
*
*
*

## Shallow network learning from raw data

> Neural network topology (architecture) :


> Number of weights :


> Test of different cases :




> analysis of this shallow network from raw data :


### Shallow network from features of the input data

> Neural network topology (architecture) :


> Number of weights :


> Test of different cases :




> analysis of this shallow network from features :


### CNN neural network for digit recognition

> Neural network topology (architecture) :


> Number of weights :


> Test of different cases :



> analysis of this CNN for digit recognition :


## CNN and their weights

> The CNNs models are deeper (have more layers), do they have more weights than the
> shallow ones? explain with one example.


## CNN for chest X-ray pneumonia recognition

> Train a CNN for the chest x-ray pneumonia recognition. In order to do so, complete the
> code to reproduce the architecture plotted in the notebook. Present the confusion matrix,
> accuracy and F1-score of the validation and test datasets and discuss your results.