import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from sklearn.model_selection import KFold, train_test_split
from utils import get_logger
from easydict import EasyDict

from causalml.inference.meta import BaseSRegressor
# from lightgbm import LGBMRegressor
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score
from sklearn.ensemble import RandomForestRegressor
from causalml.inference.meta import LRSRegressor
from causalml.propensity import compute_propensity_score
from causalml.match import NearestNeighborMatch
from causalml.inference.meta import XGBTRegressor, MLPTRegressor
from causalml.inference.meta import BaseXRegressor, BaseRRegressor, BaseSRegressor, BaseTRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import torch

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import entropy
import warnings
import logging

from causalml.inference.meta import BaseXRegressor, BaseRRegressor, BaseSRegressor, BaseTRegressor
from causalml.inference.torch import CEVAE
from causalml.propensity import ElasticNetPropensityModel
from causalml.metrics import *
from causalml.dataset import simulate_hidden_confounder


plt.style.use('fivethirtyeight')
sns.set_palette('Paired')
plt.rcParams['figure.figsize'] = (12,8)




np.random.seed(0)
torch.manual_seed(0)
exp_id = "Transformer_ATE"
output_dir = os.path.join("./results", exp_id)
logger = get_logger(output_dir, exp_id)

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# read the data files
T = np.load('./data/T.npy') #  (4338, 1) 363: 1
X = np.load('./data/X.npy') #  (4338, 10, 1437)
Y = np.load('./data/y.npy') #  (4338, 1)

print("T", T.sum())

X = torch.tensor(X, dtype=torch.float32).to(device)
Y = torch.tensor(Y, dtype=torch.float32).to(device)
T = torch.tensor(T, dtype=torch.float32).squeeze() 

# Normalize the feature matrix X
mean = X.mean(dim=(0,1), keepdim=True)
std = X.std(dim=(0,1), keepdim=True)
X = (X - mean) / std

X_train_val, X_test, Y_train_val, Y_test, T_train_val, T_test = train_test_split(X, Y, T, test_size=0.2, random_state=0)

args = {"hidden_size": 512,
        "num_heads": 8,
        "num_layers": 4,
        "dropout": 0.2,
        "learning_rate": 0.0001,
        "num_epochs": 1000}

args = EasyDict(args)
logger.info(args)
class TransformerModel(nn.Module):
    def __init__(self, input_size, num_heads=8, hidden_size=256, num_layers=4, dropout=0.2):
        super(TransformerModel, self).__init__()
        
        self.input_size = input_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Linear(input_size, hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.regressor = nn.Linear(hidden_size // 4, 1)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        x = self.embedding(x)  # shape: (batch_size, seq_len, hidden_size)

        # Permute for transformer input
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, hidden_size)

        # Transformer expects input of shape (seq_len, batch_size, hidden_size)
        transformer_out = self.transformer_encoder(x)  # (seq_len, batch_size, hidden_size)

        # Get the last timestep output
        hidden = transformer_out[-1, :, :]  # (batch_size, hidden_size)
        
        out = F.relu(self.fc1(hidden))
        out = self.dropout_layer(out)
        out = F.relu(self.fc2(out))
        out = self.dropout_layer(out)
        out = self.regressor(out)
        return out.squeeze(1)  # Return shape should match target shape (batch_size,)

    def get_features(self, x):
        x = self.embedding(x) 
        x = x.permute(1, 0, 2)
        transformer_out = self.transformer_encoder(x) 
        hidden = transformer_out[-1, :, :]  
        return hidden.squeeze(1) 



def eval_epoch(model, criterion, X_in, Y_in):
    model.eval()
    with torch.no_grad():
        predictions = model(X_in).squeeze()
        loss = criterion(predictions, Y_in.squeeze())

        auc = roc_auc_score(Y_in.cpu(), predictions.cpu())
        accuracy = accuracy_score(Y_in.cpu(), (predictions > 0.5).cpu().int())
        sensitivity = recall_score(Y_in.cpu(), (predictions > 0.5).cpu().int())
    return loss, auc, accuracy, sensitivity



def main():

    # Load the best model state
    model = TransformerModel(input_size=X.shape[-1], hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout).to(device)
    criterion = nn.MSELoss()

    ckpt = "./results/Transformer/best_fold_4.pth"
    best_model_state = torch.load(ckpt)
    model.load_state_dict(best_model_state)
    test_loss, test_auc, test_accuracy, test_sensitivity = eval_epoch(model, criterion, X_test, Y_test)
    logger.info(f'Test Set AUC: {test_auc:.4f}, Accuracy: {test_accuracy:.4f}, Sensitivity: {test_sensitivity:.4f}')


#    Extract LSTM features for train and validation datasets
    model.eval()
    with torch.no_grad():
        X_feat = model.get_features(X).cpu().numpy()
        X_feat_test = model.get_features(X_test).cpu().numpy()

    # Use a Random Forest as the base learner
    # base_learner = RandomForestRegressor()
    T_np = T.squeeze().cpu().numpy()
    Y_np = Y.squeeze().cpu().numpy()

    # ate_s = meta_learner.estimate_ate(X=X_feat, treatment=T_np, y=Y_np)
    # print(ate_s)


    # learner = LRSRegressor()
    # learner = XGBTRegressor()
    # learner = BaseTRegressor(learner=XGBRegressor())
    # learner = BaseTRegressor(learner=LinearRegression())
    # learner = BaseXRegressor(learner=XGBRegressor())
    # learner = BaseXRegressor(learner=LinearRegression())
    # learner = BaseRRegressor(learner=XGBRegressor())
    # learner = BaseRRegressor(learner=LinearRegression())
    # learner = BaseSRegressor(RandomForestRegressor())

    # learner = LRSRegressor()
    # ate_s = learner.estimate_ate(X=X_feat, treatment=T_np, y=Y_np)
    # print(ate_s)
    # logger.info(ate_s)
    # logger.info('ATE estimate: {:.03f}'.format(ate_s[0][0]))
    # logger.info('ATE lower bound: {:.03f}'.format(ate_s[1][0]))
    # logger.info('ATE upper bound: {:.03f}'.format(ate_s[2][0]))


    # 计算倾向得分
    print(X_feat.shape)
    print(T_np.shape)
    propensity_score = compute_propensity_score(X_feat, T_np)
    # 使用最近邻匹配
    nn_match = NearestNeighborMatch(replace=True, ratio=1, random_state=0)
    matched_indices = nn_match.match(propensity_score, T_np)

    # 获取匹配后的数据
    X_matched = X[matched_indices['control_indices']]
    T_matched = T_np[matched_indices['treatment_indices']]
    Y_matched = Y_np[matched_indices['treatment_indices']]

    # 合并处理组和匹配后的对照组
    X_balanced = np.vstack((X_matched, X_feat[T_np == 1]))
    T_balanced = np.hstack((np.zeros_like(T_matched), np.ones(np.sum(T_np == 1))))
    Y_balanced = np.hstack((Y_matched, Y_np[T_np == 1]))

    # 初始化LRSRegressor并估计ATE
    learner = LRSRegressor()
    ate_s = learner.estimate_ate(X=X_balanced, treatment=T_balanced, y=Y_balanced)

    print(ate_s)
    print(f"ATE estimate: {ate_s[0]:.3f}")
    print(f"ATE lower bound: {ate_s[1]:.3f}")
    print(f"ATE upper bound: {ate_s[2]:.3f}")

    # learner_s = LRSRegressor()

    # Train the CausalML model on training data
    # learner_s.fit(X_feat, T.numpy(), Y.numpy())
    # learner_s.fit(X_feat_test, T_test.numpy(), Y_test.numpy())

    # Ready-to-use S-Learner using LinearRegression
    # ate_s = learner_s.estimate_ate(X=X_feat, treatment=T, y=Y, pretrain=True)
    # print(ate_s)
    # print('ATE estimate: {:.03f}'.format(ate_s[0][0]))
    # print('ATE lower bound: {:.03f}'.format(ate_s[1][0]))
    # print('ATE upper bound: {:.03f}'.format(ate_s[2][0]))


if __name__ == "__main__":
    main()
