# Pipeline for transfer learning using DeepChem and Keras layers

import tensorflow as tf 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import deepchem as dc 
from deepchem.models import GraphConvModel
from deepchem.trans.transformers import undo_transforms


"""
I. Finetuning of layers to freeze / retrain from pre-trained model

"""

def load_data(csv):
    # select the featurizer type to be ConvMolFeaturizer 
    graph_featurizer = dc.feat.graph_features.ConvMolFeaturizer()
    
    # Use CSVLoader and load according to the smiles and tasks fields 
    loader = dc.data.data_loader.CSVLoader(tasks=['gap'], 
                                        smiles_field="smiles", 
                                        id_field="ID", 
                                        featurizer=graph_featurizer)
    dataset = loader.featurize(csv)

    # Initialize transformers on dataset 
    transformers = [
        dc.trans.NormalizationTransformer(
        transform_y=True, dataset=dataset, move_mean=True)
    ]
    for transformer in transformers:
        dataset = transformer.transform(dataset)
    # 0.8 0.2 split 
    splitter = dc.splits.splitters.RandomSplitter()
    train_set, valid_set = splitter.train_test_split(dataset = dataset, frac_train = 0.8)
    print("Split of dataset:", len(train_set), len(valid_set))

    return train_set, valid_set, transformers


def get_default_metrics():
    metric_list = [dc.metrics.Metric( dc.metrics.pearson_r2_score, np.mean),
                    dc.metrics.Metric( dc.metrics.rms_score, np.mean ),
                    dc.metrics.Metric( dc.metrics.mae_score, np.mean )]
    return metric_list

def fit_with_metrics(model = None,
                    num_epochs = 100,
                    train_set = train_set,
                    valid_set = valid_set,
                    transformers = transformers):
    metrics = get_default_metrics()
    losses = []
    all_metrics = []
    for i in range(num_epochs):
        loss = model.fit(train_set, 
                        nb_epoch = 1)
        print(f"Epoch {i} with loss: {loss}")
        losses.append(loss)
        print("Validation metrics")
        results = model.evaluate(valid_set, metrics, transformers)
        all_metrics.append(results)
        print(results)

    plt.ylabel("Dest Model Loss")
    plt.xlabel("Epoch")
    x = range(num_epochs)
    y = losses
    plt.plot(x,y)
    final_metrics = model.evaluate(valid_set, metrics, transformers)
    return losses, final_metrics, all_metrics, plt


def finetune_dest_model(model_dir = "models", 
                        source_model = None,
                        csv = "input_data.csv",
                        include_top = False,
                        num_epochs = 100):

    dest_model = GraphConvModel(n_tasks = 1,
                                graph_conv_layers = [128, 128],
                                dense_layer_size = 512,
                                dropout = 0, 
                                mode = 'regression',
                                learning_rate = 0.001,
                                batch_size = 8,
                                model_dir = model_dir)

    dest_model.load_from_pretrained(source_model = source_model,
                                   assignment_map = None,
                                   value_map = None,
                                   include_top = include_top)

    train_set, valid_set, transformers = load_data(csv)

    tune_layers_index = []
    all_layers = dest_model.model.layers
    for layer in all_layers:
        ind = all_layers.index(layer)
        namelist = layer.name.split("_")
        if "batch" and "normalization" in namelist:
            tune_layers_index.append(ind)
        elif "graph" and "conv" in namelist:
            tune_layers_index.append(ind)
        elif "dense" in namelist: 
            tune_layers_index.append(ind)

    all_model_losses = []
    all_model_metrics = []
    all_model_final_metrics = []
    all_model_plots = []
    all_models = []
    for freeze_till in tune_layers_index:
        print("-----------------------------")
        # iterate through all possible index to freeze until and fit model for each iteration 
        print(f"RUNNING ITERATION {tune_layers_index.index(freeze_till)} / {len(tune_layers_index)}")
        for layer in all_layers:
            if all_layers.index(layer) < freeze_till:
                layer.trainable = False
            else:
                layer.trainable = True
        
        print(f"Froze layers till {dest_model.model.layers[freeze_till - 1]}")
        print(f"Training layers starting from {dest_model.model.layers[freeze_till]}")
        print(f"Trainable layers: {len([layer for layer in dest_model.model.layers if layer.trainable])} - {[layer for layer in dest_model.model.layers if layer.trainable]}")
        print("-----------------------------")

        current_losses, final_metrics, current_metrics, plt = fit_with_metrics(dest_model, 
                                                                                num_epochs = num_epochs,
                                                                                train_set = train_set,
                                                                                valid_set = valid_set,
                                                                                transformers = transformers)
        all_model_losses.append(current_losses)
        all_model_final_metrics.append(final_metrics)       
        all_model_metrics.append(current_metrics)
        all_model_plots.append(plt)
        all_models.append(dest_model)
        print("Fitting completed!")
        print(f"Final metrics for this model: {current_metrics}")

    return all_models, all_model_losses, all_model_final_metrics, all_model_metrics, all_model_plots


"""
II. Early stopping based on ChemNet
"""

# Early stopping based on loss and based on selected metric from a list of metrics = MAE 
def compute_loss_on_valid(model, valid_set):
    loss_fn = model._loss_fn
    outputs = model.predict(valid_set, transformers = []) # no transformers in prediction
    loss_tensor = loss_fn([outputs], [valid_set.y], weights = [valid_set.w])
    loss = model.session.run(loss_tensor)
    return loss 

def fit_with_early_stopping(model = None,
                            num_cycles = 10, 
                            early_stopping_epochs = 10,
                            train_set = train_set,
                            valid_set = valid_set,
                            transformers = transformers):
    # early_stopping_epochs = Number of epochs to test early stopping after
    # max is num_cycles * early_stopping_epochs = 100

    loss_old = compute_loss_on_valid(model, valid_set)

    for rep_num in range(num_cycles):
        losses, final_metrics, all_metrics, plt = fit_with_metrics(model, 
                                                                num_epochs = early_stopping_epochs, 
                                                                train_set = train_set,
                                                                valid_set = valid_set,
                                                                transformers = transformers)
        loss_new = compute_loss_on_valid(model, valid_set)

        if loss_new > loss_old:
            print("No improvement in validation loss. Early stopping now.")
            break 
    return


