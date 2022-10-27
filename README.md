# Incremental-Learning-Research

This is my research for Class-Incremental learning

### Research background

Catastrophic forgetting is the key role in incremental learning, since the model gains poor quality for old classes than new classes with continuously incoming tasks and less storage. In order to solve the balance between model plasticity and stability.

### How to run

please install python3.x and pytorch(not need other packages).

please run main.py to train the model.

### Model for DCBIL

![1](/image/1.jpg)

  DCBIL structure diagram. After base model is trained, we modify it with discrimination correction; stochastic perturbation probability synergy is performed according to the output probability.

### Experiment Result

#### Accuracy graph

![image-20221027145500123](/image/2.jpg)

#### Confusion Matrix

![image-20221027145828831](/image/3.jpg)
