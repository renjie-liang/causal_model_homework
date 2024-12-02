import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from utils import get_logger
from easydict import EasyDict

from causalml.inference.meta import BaseSRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score
from sklearn.ensemble import RandomForestRegressor

# from causalml.inference.meta import LRSRegressor
# from sklearn.ensemble import RandomForestRegressor
# from scipy.stats import permutation_test

# from causalml.inference.meta import LRSRegressor
# from causalml.inference.meta import XGBTRegressor, MLPTRegressor
# from causalml.inference.meta import BaseXRegressor, BaseRRegressor, BaseSRegressor, BaseTRegressor
# from causalml.inference.tf import DragonNet
# from causalml.match import NearestNeighborMatch, MatchOptimizer, create_table_one
# from causalml.propensity import ElasticNetPropensityModel
# from causalml.dataset.regression import *
# from causalml.metrics import *


np.random.seed(0)
torch.manual_seed(0)
exp_id = "demo"
output_dir = os.path.join("./results", exp_id)
logger = get_logger(output_dir, exp_id)


# read the data files
T = np.load('./data/T.npy') #  (4338, 1)
X = np.load('./data/X.npy') #  (4338, 10, 1437)
Y = np.load('./data/y.npy') #  (4338, 1)

X = torch.tensor(X, dtype=torch.float32)
T = torch.tensor(T, dtype=torch.float32).squeeze()
Y = torch.tensor(Y, dtype=torch.float32)
print(Y.sum(), len(Y))

# Normalize the feature matrix X
mean = X.mean(dim=(0,1), keepdim=True)
std = X.std(dim=(0,1), keepdim=True)
X = (X - mean) / std


X_train_val, X_test, Y_train_val, Y_test, T_train_val, T_test = train_test_split(X, Y, T, test_size=0.2, random_state=0)

args = {"hidden_size": 128,
        "num_layers": 8,
        "dropout": 0.2,
        "learning_rate": 0.0001}
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
        return out
    
    def get_features(self, x):
        _, (hidden, _) = self.lstm(x)
        return hidden[-1]  # Take the last layer's hidden state


def eval_epoch(model, criterion, X_in, Y_in):
    model.eval()
    with torch.no_grad():
        predictions = model(X_in).squeeze(1)
        loss = criterion(predictions, Y_in)

        # Y_in_np = Y_in.numpy()
        auc = roc_auc_score(Y_in, predictions)
        accuracy = accuracy_score(Y_in, (predictions > 0.5).int())
        sensitivity = recall_score(Y_in, (predictions > 0.5).int())
    return loss, auc, accuracy, sensitivity
def main():
    num_epochs = 1000
    # K-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    fold_results = []
    ate_results = []
    best_val_acc = 0
    best_epoch = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_val)):
        print(f'Fold {fold + 1}')
        
        # Split data
        X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
        Y_train, Y_val = Y_train_val[train_idx], Y_train_val[val_idx]
        T_train, T_val = T_train_val[train_idx], T_train_val[val_idx]
        
        # Define loss function and optimizer
        model = LSTMModel(input_size=X.shape[-1], hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        # Training loop
        for epoch in range(num_epochs): 
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, Y_train)
            loss.backward()
            optimizer.step()
            
            # Validation loop
            val_loss, val_auc, val_accuracy, val_sensitivity = eval_epoch(model, criterion, X_val, Y_val)

            # Check if this is the best epoch
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_epoch =  [fold, epoch]
                best_model_state = model.state_dict()
                torch.save(model.state_dict(), f"{output_dir}/best_fold_{fold+1}.pth")

            logger.info(f'Fold {fold}, Epoch [{epoch+1}|{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
            logger.info(f'Val Set AUC: {val_auc:.4f}, Accuracy: {val_accuracy:.4f}, Sensitivity: {val_sensitivity:.4f}')
            logger.info("")
        # fold_results.append(best_val_loss)



    # Load the best model state

    # model = LSTMModel(input_size=X.shape[-1], hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout)
    # ckpt = "./results/demo/best_fold_1.pth"
    # best_model_state = torch.load(ckpt)
    model.load_state_dict(best_model_state)


    test_loss, test_auc, test_accuracy, test_sensitivity = eval_epoch(model, criterion, X_test, Y_test)
    print(f'Test Set AUC: {test_auc:.4f}, Accuracy: {test_accuracy:.4f}, Sensitivity: {test_sensitivity:.4f}')



# Extract LSTM features for train and validation datasets
# model.eval()
# with torch.no_grad():
#     X_feat = model.get_features(X)
#     # X_feat_train = model.get_features(X_train).numpy()
#     # X_feat_val = model.get_features(X_val).numpy()

# X_feat_np = X_feat.squeeze().numpy()
# X_np = X.squeeze().numpy()
# T_np = T.squeeze().numpy()
# Y_np = Y.squeeze().numpy()
# s_learner = BaseSRegressor(RandomForestRegressor())
# s_ate = s_learner.estimate_ate(X_feat_np, T_np, Y_np)[0]
# s_ite = s_learner.fit_predict(X_feat_np, T_np, Y_np)

# print("ATE:", s_ate)
# print("ITE:", s_ite)
# print()


# t_learner = BaseTRegressor(LGBMRegressor())
# t_ate = t_learner.estimate_ate(X, treatment, y)[0][0]
# t_ite = t_learner.fit_predict(X, treatment, y)

# x_learner = BaseXRegressor(LGBMRegressor())
# x_ate = x_learner.estimate_ate(X, treatment, y, p)[0][0]
# x_ite = x_learner.fit_predict(X, treatment, y, p)

# r_learner = BaseRRegressor(LGBMRegressor())
# r_ate = r_learner.estimate_ate(X, treatment, y, p)[0][0]
# r_ite = r_learner.fit_predict(X, treatment, y, p)

   # Extract LSTM features for train and validation datasets
    # model.eval()
    # with torch.no_grad():
    #     X_feat_train = model.get_features(X_train).numpy()
    #     X_feat_val = model.get_features(X_val).numpy()

    # Use a Random Forest as the base learner
    # base_learner = RandomForestRegressor()
    # meta_learner = BaseSRegressor(learner=base_learner)
    # learner_s = LRSRegressor()

    # # Train the CausalML model on training data
    # learner_s.fit(X_feat_train, T_train.numpy(), Y_train.numpy())

    # # Ready-to-use S-Learner using LinearRegression
    # ate_s = learner_s.estimate_ate(X=X_feat_val, treatment=T_val, y=Y_val, pretrain=True)
    # print(ate_s)
    # print('ATE estimate: {:.03f}'.format(ate_s[0][0]))
    # print('ATE lower bound: {:.03f}'.format(ate_s[1][0]))
    # print('ATE upper bound: {:.03f}'.format(ate_s[2][0]))

# Average performance across folds
# avg_val_loss = np.mean(fold_results)
# print(f'Average Validation Loss: {avg_val_loss:.4f}')



if __name__ == "__main__":
    main()