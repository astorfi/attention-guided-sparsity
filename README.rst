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
