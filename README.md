# deepchemTLAL
Transfer learning and active learning frameworks built together with DeepChem

## Dependencies:
- [DeepChem 2.3.0](https://github.com/deepchem/deepchem)
- rdkit 
- tensorflow 1.14

## Transfer Learning 
- With a pretrained model, this approach here freezes some layers and only trains the rest; Keras way of setting the underlying model layers to be trainable or not. 
- The ```finetune_dest_model``` function goes through possible combinations of trainable layers and fits the same dataset through it after adopting the architecture of a pretrained model
