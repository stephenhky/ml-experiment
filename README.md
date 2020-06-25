# Package `ml-experiment`

[![Build Status](https://travis-ci.org/stephenhky/ml-experiment.svg?branch=master)](https://travis-ci.org/stephenhky/ml-experiment)
[![GitHub release](https://img.shields.io/github/release/stephenhky/ml-experiment.svg?maxAge=3600)](https://github.com/stephenhky/ml-experiment/releases)
[![Documentation Status](https://readthedocs.org/projects/ml-experiment/badge/?version=latest)](https://ml-experiment.readthedocs.io/en/latest/?badge=latest)
[![Updates](https://pyup.io/repos/github/stephenhky/ml-experiment/shield.svg)](https://pyup.io/repos/github/stephenhky/ml-experiment/)
[![Python 3](https://pyup.io/repos/github/stephenhky/ml-experiment/python-3-shield.svg)](https://pyup.io/repos/github/stephenhky/ml-experiment/)

## Introduction

This Python package facilitates the fast prototyping of
machine learning model with great scalability and flexibility.

Characteristics of this package:

* Flexibility of Feature Engineering: it is convenient to define a function to 
put to feature-processing pipeline;
* Flexibility of Models: there is no restriction about whether you have to use
scikit-learn, TensorFlow, or PyTorch;
* Few Specifications on Models: user only need to worry about the `fit`
and `predict_proba`;
* Training Job Specifications: features, data locations, model specifications can
be specified in a Python dictionary or JSON, facilitating potential
MapReduce or parallelism;
* Scalability: data is stored temporarily in disks in batch
to save memory space;
* Statistics: statistical measures of the performance of the models and
their class labels are calculated;
* Cross Validation: cross validation option is available.
* Ready Adaptation to Production: data pipelines and algorithms can be adapted into
production codes with little changes.

There will be tutorials and documentations.

## News

* 06/24/2020: `0.0.6` released.
* 05/31/2020: `0.0.5` released.
* 05/12/2020: `0.0.4` released.
* 05/03/2020: `0.0.3` released.
* 04/29/2020: `0.0.2` released.
* 04/24/2020: `0.0.1` released.
