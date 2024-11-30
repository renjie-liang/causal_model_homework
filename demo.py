import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from utils import get_logger

from causalml.inference.meta import BaseSRegressor
from causalml.inference.meta import LRSRegressor
from sklearn.ensemble import RandomForestRegressor
import os
from scipy.stats import permutation_test


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

# Normalize the feature matrix X
mean = X.mean(dim=(0,1), keepdim=True)
std = X.std(dim=(0,1), keepdim=True)
X = (X - mean) / std


X_train_val, X_test, Y_train_val, Y_test, T_train_val, T_test = train_test_split(X, Y, T, test_size=0.2, random_state=0)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.5):
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


num_epochs = 10
# K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=0)
fold_results = []
ate_results = []
best_val_loss = float('inf')
best_epoch = None


for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_val)):
    print(f'Fold {fold + 1}')
    
    # Split data
    X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
    Y_train, Y_val = Y_train_val[train_idx], Y_train_val[val_idx]
    T_train, T_val = T_train_val[train_idx], T_train_val[val_idx]
    
    # Define loss function and optimizer
    model = LSTMModel(input_size=X.shape[-1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(num_epochs): 
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)
        loss.backward()
        optimizer.step()
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            predictions = model(X_val).squeeze(1)
            val_loss = criterion(predictions, Y_val)

        # Check if this is the best epoch
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch =  [fold, epoch]
            best_model_state = model.state_dict()
            torch.save(model.state_dict(), f"{output_dir}/best_fold_{fold+1}.pth")
        logger.info(f'Epoch [{epoch+1}|{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
    fold_results.append(best_val_loss)


# Load the best model state
model.load_state_dict(best_model_state)


# Calculate performance metrics on the test set
model.eval()
with torch.no_grad():
    test_predictions = model(X_test).squeeze(1)
    
    # Convert predictions and labels to numpy for metric calculation
    test_predictions_np = test_predictions.numpy()
    Y_test_np = Y_test.numpy()

# Calculate AUC, accuracy, and sensitivity
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score

auc = roc_auc_score(Y_test_np, test_predictions_np)
accuracy = accuracy_score(Y_test_np, (test_predictions_np > 0.5).astype(int))
sensitivity = recall_score(Y_test_np, (test_predictions_np > 0.5).astype(int))

print(f'Test Set AUC: {auc:.4f}')
print(f'Test Set Accuracy: {accuracy:.4f}')
print(f'Test Set Sensitivity: {sensitivity:.4f}')
# calculate the AUC, accuracy, sensitivity on test set



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
avg_val_loss = np.mean(fold_results)
print(f'Average Validation Loss: {avg_val_loss:.4f}')



