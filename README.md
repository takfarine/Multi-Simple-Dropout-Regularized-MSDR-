# 🧬 Multi Simple Dropout Regularized (MSDR)

## 🌟 Overview
This repository introduces the **Multi Simple Dropout Regularized (MSDR)** technique. By applying multiple dropout rates and associated regularizations on a single dense layer, MSDR aims to improve neural network robustness and generalization.

## 🛠 Key Components

### 1. 📉 **Varied Dropout Rates**
Defines a set of dropout rates to be applied.

### 2. 🧮 **Dynamic Regularization Calculation**
Based on the dropout rate, number of dense units, and batch size, the regularization rates (both L1 and L2) are dynamically computed.

### 3. 🔩 **Shared Dense Layer**
A common dense layer reused for each dropout rate, ensuring that weights are shared across varied regularizations.

### 4. 🎯 **Target Activation**
Post the shared dense layer, an activation function produces the output for each dropout-regularization combination.

### 5. 🔄 **Output Averaging**
The final model output is the average of outputs from all dropout-regularization pathways.

### 6. 📐 **Formulaic Relationship**
A formula that relates dropout rate, dense units, and batch size to compute regularization rates:

\[ \text{Regularization Rate} = \frac{\text{Dropout Rate}}{\text{Dense Units}} \times \text{Batch Size} \]

## 🖥 Implementation (TensorFlow)

```python
import numpy as np
import tensorflow as tf

def MSDR(input_shape, dense_units=32, batch_size=32):
    x = tf.keras.Input(shape=input_shape)
    outputs = []
    dense = []

    for p in np.linspace(0.1, 0.5, 5):
        reg_rate = p / dense_units * batch_size
        regularizer = tf.keras.regularizers.l1_l2(l1=reg_rate, l2=reg_rate)
        FC = tf.keras.layers.Dense(dense_units, activation='relu', kernel_regularizer=regularizer)
        x_ = tf.keras.layers.Dropout(p)(x)
        x_ = FC(x_)
        x_ = tf.keras.layers.Dense(1, activation='sigmoid')(x_)
        dense.append(x_)

    x = tf.keras.layers.Average()(dense)
    outputs.append(x)
    return tf.keras.Model(inputs=x, outputs=outputs)
