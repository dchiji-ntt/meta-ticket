#!/bin/bash
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar xvf cifar-100-python.tar.gz
mkdir ./__data__/cifar100
mv cifar-100-python ./__data__/cifar100/cifar-100-python
