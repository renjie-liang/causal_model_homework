import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.model_selection import KFold, train_test_split
from utils import get_logger, MetricsManager
from easydict import EasyDict
from causalml.inference.meta import BaseSRegressor
# from lightgbm import LGBMRegressor
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample
from models.Transformer import TransformerModel

np.random.seed(0)
torch.manual_seed(0)
exp_id = "Transformer_onemodel"
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

# concate the T and X
T_extended = T.unsqueeze(1).expand(-1, X.size(1))  # Shape (4338, 10)
T_extended = T_extended.unsqueeze(-1)  # Shape (4338, 10, 1)
X = torch.cat((X, T_extended), dim=2)  # Shape (4338, 10, 1438)

X = torch.nn.functional.normalize(X, dim=1)


args = {"hidden_size": 512,
        "num_heads": 8,
        "num_layers": 2,
        "dropout": 0.2,
        "learning_rate": 0.0001,
        "num_epochs": 1000}

args = EasyDict(args)
logger.info(args)


def eval_epoch(model, criterion, X_in, Y_in):
    model.eval()
    with torch.no_grad():
        predictions = model(X_in).squeeze()
        loss = criterion(predictions, Y_in.squeeze())

        auc = roc_auc_score(Y_in.cpu(), predictions.cpu())
        accuracy = accuracy_score(Y_in.cpu(), (predictions > 0.5).cpu().int())
        sensitivity = recall_score(Y_in.cpu(), (predictions > 0.5).cpu().int())
    return loss, auc, accuracy, sensitivity

# Clone the data as needed
def calculate_ate(X_val, model):
    X_val_tream = X_val.clone()
    X_val_tream[:, :, -1] = 1
    
    X_val_notream = X_val.clone()
    X_val_notream[:, :, -1] = 0

    Y_val_tream = model(X_val_tream)
    Y_val_notream = model(X_val_notream)

    val_ate = (Y_val_tream - Y_val_notream).mean().item()  # Convert from tensor to scalar for consistency
    return val_ate

# Bootstrapping to calculate confidence intervals
def bootstrap_ate(X_val, model, n_bootstrap=1000):
    ate_bootstrap = []
    for _ in range(n_bootstrap):
        X_resample = resample(X_val)  # Resample with replacement
        ate_bootstrap.append(calculate_ate(X_resample, model))
    
    lower_bound = np.percentile(ate_bootstrap, 2.5)  # 2.5th percentile
    upper_bound = np.percentile(ate_bootstrap, 97.5)  # 97.5th percentile
    return lower_bound, upper_bound

def main():
    num_epochs = args.num_epochs
    # K-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    fold_results = []
    ate_results = []
    best_val_acc = 0
    best_epoch = None
    metrics_manager = MetricsManager()

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
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
            val_loss, val_auc, val_accuracy, val_sensitivity = eval_epoch(model, criterion, X_val, Y_val)

            # Check if this is the best epoch
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_epoch =  [fold, epoch]
                best_model_state = model.state_dict()
                torch.save(model.state_dict(), f"{output_dir}/best_fold_{fold+1}.pth")

            logger.info(f'Fold {fold}, Epoch [{epoch+1}|{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
            logger.info(f'Val Set AUC: {val_auc:.4f}, Accuracy: {val_accuracy:.4f}, Sensitivity: {val_sensitivity:.4f}')
        # fold_results.append(best_val_loss)

        # Load the best model state
        model.load_state_dict(best_model_state)
        val_loss, val_auc, val_acc, val_sen = eval_epoch(model, criterion, X_val, Y_val)
        metrics_manager['auc'].add(val_auc)
        metrics_manager['acc'].add(val_acc)
        metrics_manager['sen'].add(val_sen)

        val_ate = calculate_ate(X_val, model)
        val_ate_low, val_ate_up = bootstrap_ate(X_val, model)
        metrics_manager['ate'].add(val_ate)
        metrics_manager['ate_low'].add(val_ate_low)
        metrics_manager['ate_up'].add(val_ate_up)

    # Load the best model state
    logger.info(f"Best fold {best_epoch}")
    logger.info(args)
    logger.info(f"acc: {metrics_manager['acc'].mean():.4f}±{metrics_manager['acc'].std():.4f}")
    logger.info(f"auc: {metrics_manager['auc'].mean():.4f}±{metrics_manager['auc'].std():.4f}")
    logger.info(f"sen: {metrics_manager['sen'].mean():.4f}±{metrics_manager['sen'].std():.4f}")
    logger.info(f"ate: {metrics_manager['ate'].mean():.4f}±{metrics_manager['ate'].std():.4f}")
    logger.info(f"ate_low: {metrics_manager['ate_low'].mean():.4f}±{metrics_manager['ate_low'].std():.4f}")
    logger.info(f"ate_up: {metrics_manager['ate_up'].mean():.4f}±{metrics_manager['ate_up'].std():.4f}")
    

if __name__ == "__main__":
    main()
