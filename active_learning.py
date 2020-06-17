# Pipeline for active learning using DeepChem and uncertainty in prediction based on dropout

import numpy as np 
import pandas as pd 
import tensorflow as tf
import deepchem as dc
from deepchem.models import GraphConvModel
from deepchem.data.datasets import DiskDataset

# own packages
from dataset_functions import to_dataframe, from_dataframe

# select the featurizer type to be ConvMolFeaturizer 
graph_featurizer = dc.feat.graph_features.ConvMolFeaturizer()

loader = dc.data.data_loader.CSVLoader(tasks=['gap'], 
                                    smiles_field="smiles", 
                                    id_field="ID", 
                                    featurizer=graph_featurizer)
metric = [dc.metrics.Metric(dc.metrics.mae_score, np.mean)]

model = GraphConvModel(n_tasks = 1,
                        graph_conv_layers = [128, 128],
                        dense_layer_size = 512,
                        dropout = 0.0001, 
                        # dropout must be included in every layer for uncertainty
                        mode = 'regression',
                        uncertainty = True,
                        learning_rate = 0.001,
                        batch_size = 8)


csv_list = ["set1.csv", "set2.csv", "set3.csv", "set4.csv"]
seeds = [5, 10, 12, 18]

# for generic train test split loads for initial run 
def load_and_split(csv, seed = None):
    # load csv for training and test 
    dataset = loader.featurize(csv)
    
    # transform data here 
    transformers = [
        dc.trans.NormalizationTransformer(
        transform_y=True, dataset=dataset, move_mean=True)
    ]
    for transformer in transformers:
        dataset = transformer.transform(dataset)
        
    # 8020 splits using deepchem split 
    splitter = dc.splits.RandomSplitter()
    train_set, test_set = splitter.train_test_split(dataset, 
                                                    frac_train = 0.8, 
                                                    seed = seed)
    print(f"train set size {len(train_set)} and test set size {len(test_set)}")

    return train_set, test_set, transformers


def train_and_predict(model, train_set, test_set, transformers):
    # fit model for 10 epochs first
    num_epochs = 10
    for i in range(num_epochs):
        loss = model.fit(train_set, nb_epoch = 1)
        print(f"Epoch {i} with loss: {loss}")
    
    # predictions from train set and get train MAE
    train_pred1 = model.predict(dataset = train_set, transformers = transformers)
    # print(f"train_pred1 from normal prediction = {train_pred1}")
    train_mae = model.evaluate(train_set, metric, transformers)

    # get uncertainty from train set 
    train_pred, train_pred_std = model.predict_uncertainty(train_set)
    # print(f"train_pred from uncertainty prediction = {train_pred}")
    
    # predictions from test set and get test MAE
    test_pred1 = model.predict(dataset = test_set, transformers = transformers)
    # print(f"y_pred1 from normal prediction = {test_pred1}")
    test_mae = model.evaluate(test_set, metric, transformers)
    
    # get uncertainty from test set 
    test_pred, test_pred_std = model.predict_uncertainty(test_set)
    # print(f"y_pred from uncertainty prediction = {test_pred}")

    return train_mae, test_mae


# load only, no need to split; for AL loading
def load(csv):
    # load csv for AL loop
    dataset = loader.featurize(csv)
    # transform data here 
    transformers = [
        dc.trans.NormalizationTransformer(
        transform_y=True, dataset=dataset, move_mean=True)
    ]
    for transformer in transformers:
        dataset = transformer.transform(dataset)
    return dataset, transformers

# ** not sure why predict_uncertainty does not require TRANSFORMERS??? ...
# def predict_and_select(model, X_new, select_frac, transformers):

def predict_and_select(model, new_set, select_frac):
    # first get uncertainty
    pred, pred_std = model.predict_uncertainty(new_set)
    # print(f"prediction from uncertainty prediction = {pred}")
    
    # find the molecules with the highest prediction uncertainty from the prediction 
    uncertain_select = np.argsort(pred_std.transpose())[0]
    num_select = int(np.round(len(uncertain_select) * (1.0 - select_frac)))
    # argsort here sorts from min to max --> selecting from the end of the array 
    # will give the max uncertainty!
    return uncertain_select[num_select: ]


def merge_and_split(select_idx, new_set, train_set, test_set, seed = None):
    # get df of the train set and test set first
    train_df = to_dataframe(train_set)
    test_df = to_dataframe(test_set)
    # then get df of all molecules from the new set in this current loop
    new_data_df = to_dataframe(new_set)

    # filter out the uncertain molecules from the new set 
    x_uncertain = new_data_df.X[select_idx]
    y_uncertain = new_data_df.y[select_idx]
    w_uncertain = new_data_df.w[select_idx]
    id_uncertain = new_data_df.ids[select_idx]
    # and form the uncertain df by combining the columns
    uncertain_df = pd.concat([x_uncertain, y_uncertain, w_uncertain, id_uncertain], axis=1)

    # then combine all train, test, and uncertain dfs together into 1 dataframe
    total_df = pd.concat([train_df, test_df, uncertain_df], axis = 0)
    total_df = total_df.reset_index(drop = True) # TRY: resetting index to make everything consistent if it affects?
    total_set = from_dataframe(total_df) # need to do this to make the disk dataset consistent with normally loaded disk dataset..
    final_disk_data = DiskDataset.from_numpy(X = total_set.X.transpose()[0], # to keep consistent dataset shape
                                            y = total_set.y,
                                            w = total_set.w,
                                            ids = total_set.ids)

    # finally, do 8020 random splits of the total set 
    splitter = dc.splits.RandomSplitter()
    new_tot_train, new_tot_test = splitter.train_test_split(final_disk_data, 
                                                            frac_train = 0.8, 
                                                            seed = seed)
    return new_tot_train, new_tot_test


# Active Learning Loops 

for i, csv in enumerate(csv_list):
    # First learning loop with train test set to get initial MAE
    if i == 0:
        train_set, test_set, transformers = load_and_split(csv, seed=seeds[i])
        train_mae, test_mae = train_and_predict(model, train_set, test_set, transformers)
        print("\n")
        print("Initial train set size: " + str(len(train_set)))
        print("Initial test set size: " + str(len(test_set)))
        print("Initial Train MAE: " + str(train_mae))
        print("Initial Test MAE: " + str(test_mae))
        print("\n")
    
    # All remaining loops are for active learning 
    else:
        # X_new, y_new = AL_load(csv)
        new_set, transformers = load(csv)

        # select the most uncertain molecules 
        select_idx = predict_and_select(model, new_set, 0.6)
        
        # then merge the new data selected into the dataset
        train_set, test_set = merge_and_split(select_idx, new_set, # new set from this current loop
                                                train_set, # train set from last saved train set
                                                test_set,  # test set from last saved test set
                                                seed=seeds[i])

        # finally put the new train set and test set into fitting and predictions 
        train_mae, test_mae = train_and_predict(model, train_set, test_set, transformers)
        print("\n")
        print("Current train set size: " + str(len(train_set)))
        print("Current test set size: " + str(len(test_set)))
        print(f"Loop {i} Train MAE: " + str(train_mae))
        print(f"Loop {i} Test MAE: " + str(test_mae))
        print("\n")