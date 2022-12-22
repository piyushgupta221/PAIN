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

## Citation
If you publish work that uses or refers to PAIN, please cite the following paper:

@article{liang2021ncvx,
    title={{NCVX}: {A} User-Friendly and Scalable Package for Nonconvex 
    Optimization in Machine Learning}, 
    author={Buyun Liang, Tim Mitchell, and Ju Sun},
    year={2021},
    eprint={2111.13984},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
