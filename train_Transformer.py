import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.model_selection import KFold, train_test_split
from utils import get_logger, MetricsManager, plot_auc_curves
from easydict import EasyDict
from causalml.inference.meta import BaseSRegressor
# from lightgbm import LGBMRegressor
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score
from sklearn.ensemble import RandomForestRegressor
from causalml.inference.meta import LRSRegressor
from causalml.inference.meta import BaseXRegressor, BaseRRegressor, BaseSRegressor, BaseTRegressor
from xgboost import XGBRegressor
from sklearn.metrics import roc_curve, auc


np.random.seed(0)
torch.manual_seed(0)
exp_id = "Transformer"
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
T = torch.tensor(T, dtype=torch.float32).squeeze().to(device)
Y = torch.tensor(Y, dtype=torch.float32).to(device)


X = torch.nn.functional.normalize(X, dim=1)

args = {"hidden_size": 512,
        "num_heads": 8,
        "num_layers": 2,
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
    return loss, auc, accuracy, sensitivity, predictions

def main():
    num_epochs = args.num_epochs
    # K-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    best_epoch = None
    metrics_manager = MetricsManager()
    fprs = []
    tprs = []
    aucs = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f'Fold {fold + 1}')
        best_val_acc = 0
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]
        T_train, T_val = T[train_idx], T[val_idx]
        
        # Define loss function and optimizer

        model = TransformerModel(input_size=X.shape[-1], hidden_size=args.hidden_size, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        # Training loop
        for epoch in range(num_epochs): 
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, Y_train.squeeze())
            loss.backward()
            optimizer.step()
            
            # Validation loop
            val_loss, val_auc, val_accuracy, val_sensitivity,_ = eval_epoch(model, criterion, X_val, Y_val)

            # Check if this is the best epoch
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_epoch =  [fold, epoch]
                best_model_state = model.state_dict()
                torch.save(model.state_dict(), f"{output_dir}/best_fold_{fold+1}.pth")


            logger.info(f'Fold {fold}, Epoch [{epoch+1}|{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
            logger.info(f'Val Set AUC: {val_auc:.4f}, Accuracy: {val_accuracy:.4f}, Sensitivity: {val_sensitivity:.4f}')
            logger.info("")

        # Load the best model state
        model.load_state_dict(best_model_state)
        val_loss, val_auc, val_acc, val_sen, val_predictions= eval_epoch(model, criterion, X_val, Y_val)
        metrics_manager['auc'].add(val_auc)
        metrics_manager['acc'].add(val_acc)
        metrics_manager['sen'].add(val_sen)

        #    Extract LSTM features for train and validation datasets
        model.eval()
        with torch.no_grad():
            X_train_feat = model.get_features(X_train).cpu().numpy()
            X_val_feat = model.get_features(X_val).cpu().numpy()

        # Use a Random Forest as the base learner
        # base_learner = RandomForestRegressor()
        T_train_np = T_train.squeeze().cpu().numpy()
        T_val_np = T_val.squeeze().cpu().numpy()
        Y_train_np = Y_train.squeeze().cpu().numpy()
        Y_test_np = Y_val.squeeze().cpu().numpy()

        # learner = LRSRegressor()
        # learner = BaseTRegressor(learner=XGBRegressor())
        learner = BaseSRegressor(learner=RandomForestRegressor())

        learner.fit(X=X_train_feat, treatment=T_train_np, y=Y_train_np)
        val_ate = learner.estimate_ate(X=X_val_feat, treatment=T_val_np, y=Y_test_np, pretrain=True)
        metrics_manager['ate'].add(val_ate[0])
        # metrics_manager['ate'].add(val_ate[0][0])
        # metrics_manager['ate_low'].add(val_ate[1][0])
        # metrics_manager['ate_up'].add(val_ate[2][0])
        fpr, tpr, _ = roc_curve(Y_val.cpu(), val_predictions.cpu())
        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(val_auc)

    logger.info(f"Best fold {best_epoch}")
    logger.info(args)

    logger.info(f"acc: {metrics_manager['acc'].mean():.4f}±{metrics_manager['acc'].std():.4f}")
    logger.info(f"auc: {metrics_manager['auc'].mean():.4f}±{metrics_manager['auc'].std():.4f}")
    logger.info(f"sen: {metrics_manager['sen'].mean():.4f}±{metrics_manager['sen'].std():.4f}")
    logger.info(f"ate: {metrics_manager['ate'].mean():.4f}±{metrics_manager['ate'].std():.4f}")
    # logger.info(f"ate_low: {metrics_manager['ate_low'].mean():.4f}±{metrics_manager['ate_low'].std():.4f}")
    # logger.info(f"ate_up: {metrics_manager['ate_up'].mean():.4f}±{metrics_manager['ate_up'].std():.4f}")
    plot_auc_curves(fprs, tprs, aucs, f"{exp_id}_auc.png")
    

if __name__ == "__main__":
    main()
