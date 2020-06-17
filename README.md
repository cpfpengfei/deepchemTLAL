## Transfer Learning and Active Learning frameworks built on top of DeepChem

## Key Dependencies:
- [DeepChem 2.3.0](https://github.com/deepchem/deepchem)
- [RDkit](https://www.rdkit.org/) 
- [Tensorflow 1.14](https://www.tensorflow.org/install/pip)

## Transfer Learning 
- With a pretrained model, this approach here freezes some layers and only trains the rest; Keras way of setting the underlying model layers to be trainable or not. 
- The ```finetune_dest_model``` function goes through possible combinations of trainable layers and fits the same dataset through it after adopting the architecture of a pretrained model

## Active Learning (WIP)
- Base model is built using uncertainty in predictions via the ```predict_uncertainty``` method in DeepChem's GraphConvModel that is derived from allocating dropout to GraphConv and Dense layers. The active learning framework is inspired from this notebook on [Gaussian Process Regressor](https://github.com/wood-b/molecular_conformations_ML/blob/master/notebooks/conformer_active_gpr.ipynb).
