import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from utils import get_logger
from easydict import EasyDict

from causalml.inference.meta import BaseSRegressor
# from lightgbm import LGBMRegressor
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score
from sklearn.ensemble import RandomForestRegressor
from causalml.inference.meta import LRSRegressor

from causalml.inference.meta import XGBTRegressor, MLPTRegressor
from causalml.inference.meta import BaseXRegressor, BaseRRegressor, BaseSRegressor, BaseTRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

np.random.seed(0)
torch.manual_seed(0)
exp_id = "LSTM_ATE"
output_dir = os.path.join("./results", exp_id)
logger = get_logger(output_dir, exp_id)

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# read the data files
T = np.load('./data/T.npy') #  (4338, 1)
X = np.load('./data/X.npy') #  (4338, 10, 1437)
Y = np.load('./data/y.npy') #  (4338, 1)

X = torch.tensor(X, dtype=torch.float32).to(device)
Y = torch.tensor(Y, dtype=torch.float32).to(device)
T = torch.tensor(T, dtype=torch.float32).squeeze() 

# Normalize the feature matrix X
mean = X.mean(dim=(0,1), keepdim=True)
std = X.std(dim=(0,1), keepdim=True)
X = (X - mean) / std

X_train_val, X_test, Y_train_val, Y_test, T_train_val, T_test = train_test_split(X, Y, T, test_size=0.2, random_state=0)

args = {"hidden_size": 512,
        "num_layers": 2,
        "dropout": 0.2,
        "learning_rate": 0.0001,
        "num_epochs": 1000}
args = EasyDict(args)
logger.info(args)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.regressor = nn.Linear(hidden_size // 4, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        hidden = hidden[-1]  # Take the last layer's hidden state
        out = self.relu(self.fc1(hidden))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.regressor(out)
        return out.squeeze(1) 
    
    def get_features(self, x):
        _, (hidden, _) = self.lstm(x)
        return hidden[-1]  # Take the last layer's hidden state

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
    model = LSTMModel(input_size=X.shape[-1], hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout).to(device)
    criterion = nn.MSELoss()

    ckpt = "./results/LSTM_ckpt/best_fold_4.pth"
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


    # meta_learner = BaseSRegressor(RandomForestRegressor())
    # ate_s = meta_learner.estimate_ate(X=X_feat, treatment=T_np, y=Y_np)
    # print(ate_s)


    # learner = LRSRegressor()
    # learner = XGBTRegressor()
    # learner = BaseTRegressor(learner=XGBRegressor())
    # learner = BaseTRegressor(learner=LinearRegression())
    # learner = BaseXRegressor(learner=XGBRegressor())
    # learner = BaseXRegressor(learner=LinearRegression())
    # learner = BaseRRegressor(learner=XGBRegressor())
    learner = BaseRRegressor(learner=LinearRegression())

    ate_s = learner.estimate_ate(X=X_feat, treatment=T_np, y=Y_np)
    logger.info(ate_s)
    logger.info('ATE estimate: {:.03f}'.format(ate_s[0][0]))
    logger.info('ATE lower bound: {:.03f}'.format(ate_s[1][0]))
    logger.info('ATE upper bound: {:.03f}'.format(ate_s[2][0]))




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
