===================================================================================
Attention-Based Guided Structured Sparsity of Deep Convolutional Neural Networks
===================================================================================

This repository contains the code developed by TensorFlow_ for the following paper:


| `Attention-Based Guided Structured Sparsity of Deep Convolutional Neural Networks`_,
| by: `Amirsina Torfi`_ and Rouzbeh Asghari Shirvani

.. _Attention-Based Guided Structured Sparsity of Deep Convolutional Neural Networks: https://openreview.net/pdf?id=S1dGIXVUz
.. _TensorFlow: https://www.tensorflow.org/
.. _Amirsina Torfi: https://astorfi.github.io/


-----------------
Goal and Outcome
-----------------

Network pruning is aimed at imposing sparsity in a neural network architecture
by increasing the portion of zero-valued weights for reducing its size energyefficiency
consideration and increasing evaluation speed. In most of the conducted
research efforts, the sparsity is enforced for network pruning without any attention
to the internal network characteristics such as unbalanced outputs of the neurons or
more specifically the distribution of the weights and outputs of the neurons. That
may cause severe accuracy drop due to uncontrolled sparsity. In this work, we
propose an attention mechanism that simultaneously controls the sparsity intensity
and supervised network pruning by keeping important information bottlenecks of
the network to be active. On CIFAR-10, *the proposed method outperforms the
best baseline method by 6% and reduced the accuracy drop by 2.6× at the same
level of sparsity.*

-------------------
Scope of the works
-------------------

In this work, we proposed a controller mechanism for network pruning with the goal of (1) model
compression for having few active parameters by enforcing group sparsity, (2) preventing the accuracy
drop by controlling the sparsity of the network using an additional loss function by forcing a
portion of the output neurons to stay alive in each layer of the network, and (3) capability of being
incorporated for any layer type


.. |im| image:: _img/varianceloss.gif

|im|


-------------
Requirements
-------------

~~~~~~~~~~~
TensorFLow
~~~~~~~~~~~

This code is written in Python and requires **TensorFlow** as the framework. For installation on *Ubuntu*, installing
TensorFlow with *GPU support* can be as follows:

.. code:: shell

    sudo apt-get install python3-pip python3-dev # for Python 3.n
    pip3 install tensorflow-gpu

Please refer to `Official TensorFLow installation guideline`_ for further details considering your specific system architecture.

.. _Official TensorFLow installation guideline: https://openreview.net/pdf?id=S1dGIXVUz
