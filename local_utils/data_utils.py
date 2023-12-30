
import pandas as pd
import csv
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import torch
import json
import seaborn as sns

#REPRODUCIBILIY
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def make_path(pathdir):
    if not os.path.exists(pathdir):
        os.mkdir(pathdir)

#VISUALIZING THE FREQUENCY DETAILS OF THE DATA
def viz_data(full_data):
    
    class_count = dict(full_data["BROWSE_NODE_ID"].value_counts())
    counts = list(class_count.values())

    class_bins = {
        "top 50" : sum(counts[:50]),
        "50 - 100" : sum(counts[50:100]),
        "100 - 150" : sum(counts[100:150]),
        "150 - 200" : sum(counts[150:200]),
        "200 - 250" : sum(counts[200:250])
    }
    colors = sns.color_palette('pastel')[0:5]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Frequency Visualization', fontsize =15)

    class_count = dict(full_data["BROWSE_NODE_ID"].value_counts())
    counts = list(class_count.values())
    ax1.plot(counts)
    ax1.set_xlabel ("class indices reverse sorted by frequency", fontsize =15)
    ax1.set_ylabel("frequency", fontsize =15)
    ax1.set_title("class frequency plot", fontsize =15)

    ax2.pie(list(class_bins.values()),
            labels = list(class_bins.keys()), 
            colors = colors, autopct='%1.1f%%')
    ax2.legend(labels = list(class_bins.keys()), title="Legend", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    ax2.set_title("Train data class bins composition", fontsize =15)

    plt.tight_layout()
    plt.show()

# CAPPING THE MAX SAMPLES PER CLASS, AND KEEPING THE MOST FREQUENT CLASSES 
def get_protov2_df(df, greater_than = 900, max_samples_per_class = 10100):
    final_df = 0
    protov2_df = pd.DataFrame()
    class_samples = df["BROWSE_NODE_ID"].value_counts()
    protov2_class_list = class_samples[class_samples.gt(greater_than)].index
    protov2_num_classes = len(protov2_class_list)
    for ind, class_id in enumerate(protov2_class_list):
        if class_samples[class_id] > max_samples_per_class:
            class_df = df[df["BROWSE_NODE_ID"] == class_id]
            percent_samples = max_samples_per_class/len(class_df)
            protov2_df = pd.concat([protov2_df, class_df.sample(frac= percent_samples, random_state = 200)])
        else:
            class_df = df[df["BROWSE_NODE_ID"] == class_id]
            percent_samples = 1
            protov2_df = pd.concat([protov2_df, class_df])
        if ind%10 == 0 and ind!= 0:    
            print(f"{ind}/{protov2_num_classes} : class id {class_id}, % of available samples : {percent_samples}, total samples added : {percent_samples * len(class_df)}")    
        if ind%50 == 0 and ind!= 0:
            print(f"\n################# samples in protov2 df : {len(protov2_df)} ##################\n")
    
    print(f"\n_________________ TOTAL SAMPLES in protov2 df : {len(protov2_df)} __________________\n")
    print(f"\n_________________ TOTAL CLASSES in protov2 df : {protov2_num_classes} __________________\n")
    print(f"\n_________________ PERCENT of total data used : {(len(protov2_df)/len(df))*100}%")
    return protov2_df.sample(frac= 1, random_state = 200)


#TEST TRAIN SPLIT
def get_data_splits(temp_df):
    train_df = temp_df.sample(frac=0.98, random_state = 200)
    test_df = temp_df.drop(train_df.index)
    
    class_samples = test_df["BROWSE_NODE_ID"].value_counts()
    class_list = list(class_samples.index)
    train_df = train_df[train_df['BROWSE_NODE_ID'].isin(class_list)]
    
    final_test_df = pd.DataFrame()
    
    for ind, class_id in enumerate(class_list):
        if class_samples[class_id] > 25:
            class_df = test_df[test_df["BROWSE_NODE_ID"] == class_id]
            percent_samples = 25/len(class_df)
            final_test_df = pd.concat([final_test_df, class_df.sample(frac= percent_samples, random_state = 200)])
        else:
            class_df = test_df[test_df["BROWSE_NODE_ID"] == class_id]
            percent_samples = 1
            final_test_df = pd.concat([final_test_df, class_df])
        #if ind%10 == 0 and ind!= 0:    
        #    print(f"{ind}/{protov2_num_classes} : class id {class_id}, % of available samples : {percent_samples}, total samples added : {percent_samples * len(class_df)}")    
    
    return (train_df, final_test_df)