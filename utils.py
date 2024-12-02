import time
import logging
import os
from collections import defaultdict


def get_logger(dir, tile):
    os.makedirs(dir, exist_ok=True)
    log_file = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(dir, "{}_{}.log".format(log_file, tile))

    logger = logging.getLogger()
    logger.setLevel('DEBUG')
    BASIC_FORMAT = "%(levelname)s:%(message)s"
    # DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)

    fhlr = logging.FileHandler(log_file) 
    fhlr.setFormatter(formatter)
    fhlr.setLevel('INFO') 

    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    return logger



class Meter:
    def __init__(self):
        self.reset()

    def add(self, value):
        self.values.append(value)

    def mean(self):
        return sum(self.values) / len(self.values) if self.values else 0.0

    def std(self):
        if len(self.values) < 2:
            return 0.0
        mean_val = self.mean()
        variance = sum((x - mean_val) ** 2 for x in self.values) / (len(self.values) - 1)
        return variance ** 0.5

    def reset(self):
        self.values = []

class MetricsManager:
    def __init__(self):
        self.meters = defaultdict(Meter)

    def __getitem__(self, key):
        return self.meters[key]

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
def normalize_no_dummy(X):
    # Device of the input tensor
    device = X.device
    
    # Assuming dummy variables are binary (0 or 1)
    is_binary = lambda col: torch.all((col == 0) | (col == 1)).item()

    samples, time_steps, features = X.shape

    # Identify dummy and continuous columns
    binary_columns = [i for i in range(features) if is_binary(X[:, :, i])]
    continuous_columns = [i for i in range(features) if i not in binary_columns]

    # Normalize continuous features along the feature axis
    X_continuous = X[:, :, continuous_columns].clone()
    X_dummy = X[:, :, binary_columns].clone()

    # Reshape continuous part for normalization
    X_continuous_reshaped = X_continuous.view(-1, len(continuous_columns)).cpu().numpy()

    # Normalize continuous features using StandardScaler
    scaler = StandardScaler()
    X_continuous_normalized = scaler.fit_transform(X_continuous_reshaped)

    # Convert back to tensor and reshape to original dimensions
    X_continuous_normalized = torch.tensor(X_continuous_normalized, dtype=torch.float32).view(samples, time_steps, len(continuous_columns)).to(device)

    # Reconstruct the final processed tensor by combining normalized continuous and original dummy features
    X_processed = torch.zeros_like(X)
    X_processed[:, :, continuous_columns] = X_continuous_normalized
    X_processed[:, :, binary_columns] = X_dummy

    return X_processed

import matplotlib.pyplot as plt
def plot_auc_curves(fprs, tprs, aucs, save_name):
    plt.figure(figsize=(10, 8))  # Set the figure size
    lw = 2
    for i in range(len(fprs)):
        plt.plot(fprs[i], tprs[i], lw=lw, label=f'Fold {i+1} ROC curve (area = {aucs[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for 5-fold Cross Validation')
    plt.legend(loc="lower right")
    plt.savefig(save_name)



