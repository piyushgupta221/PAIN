# Towards Physically Adversarial Intelligent Networks (PAINs) for Safer Self-Driving

This code has been tested with CARLA 0.9.6

To use the code or trained models, follow the following steps:
- Install CARLA 0.9.6
- Copy the code in src folder in the CARLA directory with CARLA's PythonAPI folder
- run worldloader.py

## src/training
This folder contains the code to train the models, or to use trained models. Set the flags Train_network and use_model in Q_network.py according to your application.

## src/models
This folder contains the models for the trained adversary, baseline protagonist, protagonist model1, and protagonist model2. For details refer to the paper- Towards Physically Adversarial Intelligent Networks (PAINs) for Safer Self-Driving

## src/Performance_evaluation
This folder contains the code for experiments 1-4 detailed in the manuscript - Towards Physically Adversarial Intelligent Networks (PAINs) for Safer Self-Driving

## BibTeX Citation

If you use or refer to PAIN in a scientific publication, we would appreciate using the following citation to our [paper](https://ieeexplore.ieee.org/abstract/document/9991836):

P. Gupta, D. Coleman and J. E. Siegel, "Towards Physically Adversarial Intelligent Networks (PAINs) for Safer Self-Driving," in IEEE Control Systems Letters, doi: 10.1109/LCSYS.2022.3230085.

```
@ARTICLE{9991836,
  author={Gupta, Piyush and Coleman, Demetris and Siegel, Joshua E.},
  journal={IEEE Control Systems Letters}, 
  title={Towards Physically Adversarial Intelligent Networks (PAINs) for Safer Self-Driving}, 
  year={2023},
  volume={7},
  number={},
  pages={1063-1068},
  doi={10.1109/LCSYS.2022.3230085}}

```
