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

# TODO control the responses and reformat everything !!!!!!



> What is the learning algorithm being used to optimize the weights of the neural networks?

The learning algorithm used in the various experiences are :

Notebook A
Learning Algorithm: Stochastic Gradient Descent (SGD)
A commonly used optimization algorithm for training neural networks. It updates the model's weights by computing gradients of the loss function with respect to the weights and then applying a step in the opposite direction.

Notebook B
Learning Algorithm: Adam (Adaptive Moment Estimation)
Adam combines the benefits of Adaptive Gradient Algorithm (AdaGrad) and Root Mean Square Propagation (RMSProp). It uses adaptive learning rates and momentum to optimize the weights, resulting in faster convergence.

Notebook C
Learning Algorithm: RMSProp
RMSProp is an adaptive learning algorithm that adjusts the learning rate based on the average of recent gradients. This helps to mitigate the oscillations and stabilizes the training process.

> What are the parameters (arguments) being used by that algorithm?

The following parameters had been used for each algorithms :



Notebook A
RMSprop Parameters
lr (Learning Rate): Controls the size of the steps taken during optimization. Commonly between 0.001 and 0.01.
rho: Decay rate for the moving average of squared gradients. Usually around 0.9.
eps (Epsilon): Small constant to avoid division by zero; typically 1e-8 or similar.
weight_decay: Regularization parameter; often 0 or a small value like 1e-4.
momentum: Adds momentum to the optimizer; typical values are 0 or 0.9.
Loss Function
Cross-Entropy Loss: Typically used for classification tasks. Measures dissimilarity between predicted output and actual target.


Notebook B
RMSprop Parameters
Similar to Notebook A, with some variations:
lr: Typically 0.001 to 0.01.
rho: Generally 0.9.
eps: Around 1e-8.
weight_decay: Often 0 or another small value.
momentum: Typically 0 or 0.9.
Loss Function
If dealing with a regression task, a common choice is:
Mean Squared Error (MSE): Calculates the average of squared differences between predicted and actual values.


Notebook C
RMSprop Parameters
Usually consistent with A and B, with possible variations for specific tasks:
lr: Often between 0.001 and 0.01.
rho: Generally around 0.9.
eps: Usually 1e-8 or similar.
weight_decay: Typically 0.
momentum: Often 0.
Loss Function
In mixed tasks or where outliers are a concern, a common choice is:
Huber Loss: A combination of MSE and Mean Absolute Error (MAE), providing robustness to outliers with smoother gradients.



> What loss function is being used ?

The loss functions are the following :
Notebook A
Loss Function: Cross-Entropy Loss
This loss function is typically used for classification tasks. It measures the difference between predicted class probabilities and the actual class labels.

Notebook B
Loss Function: Mean Squared Error (MSE)
This loss function is commonly used for regression tasks. It calculates the average of the squared differences between the predicted values and the actual target values.

Notebook C
Loss Function: Huber Loss
This loss function is a combination of Mean Squared Error (MSE) and Mean Absolute Error (MAE). It is robust to outliers while maintaining smooth gradients for optimization.

> Please, give the equation(s)

* Notebook A: Stochastic Gradient Descent (SGD)
$\\w_{t+1} = w_t - \eta \cdot \nabla L(w_t)\\$

* Notebook B: Adam (Adaptive Moment Estimation)

Update biased first moment (mean) and second moment (uncentered variance) estimates
$\\\begin{align}
m_{t+1} = \beta_1 \cdot m_t + (1 - \beta_1) \cdot g_t
\end{align}\\$

$\\\begin{align}
v_{t+1} = \beta_2 \cdot v_t + (1 - \beta_2) \cdot g_t^2
\end{align}\\$

Compute bias-corrected first and second moment estimates
$\\\begin{align}
\hat{m}_{t+1} = \frac{m_{t+1}}{1 - \beta_1^{t+1}}
\end{align}\\$

$\\\begin{align}
\hat{v}_{t+1} = \frac{v_{t+1}}{1 - \beta_2^{t+1}}
\end{align}\\$

Update weights
$\\\begin{align}
w_{t+1} = w_t - \eta \cdot \frac{\hat{m}_{t+1}}{(\sqrt{\hat{v}_{t+1}} + \epsilon)}
\end{align}\\$

* Notebook C: RMSProp

Update squared gradients' moving average
$\\\begin{align}
v_{t+1} = \beta \cdot v_t + (1 - \beta) \cdot g_t^2
\end{align}\\$

Update weights
$\\\begin{align}
w_{t+1} = w_t - \eta \cdot \frac{g_t}{\sqrt{v_{t+1}} + \epsilon}
\end{align}\\$



## Shallow network learning from raw data

> Neural network topology (architecture) :

The inputs in our neural network are each of the pixels of the MNNIST dataset I.E. 28*28 = 784 entry neurons.

We have 60k entries for the train set and 10k for the test set. Each record are split into 10 classes. Each of them being a representation of a digit 0 to 9.



> Number of weights :


> Test of different cases :

After several experiments we decided to go for the following topology with a _relu_ activation on the output layer:
![image](assets/raw_topology.png)

This topology allows us to reach an accuracy of about 98.3%

We tested with :

    - no dropout layer  -> 98 %
    - with 300 neurons in the hidden layer -> 97.8 %
    - with dropout 0.3 -> 98 %
    - with tanh activation on the output layer -> 97.7 %
    - with 240 neurons in the hidden layer -> 98.2 %

Those trials showed us that here, the simpler the architecture is the better it performs.






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