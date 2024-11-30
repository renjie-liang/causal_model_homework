srun --partition=gpu --gres=gpu:1 --nodes=1 --cpus-per-task=16 --mem=100gb --time=02:00:00 --account=bianjiang --qos=bianjiang --reservation=bianjiang --pty bash -i


```
INFO:{'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2, 'learning_rate': 0.0001, 'num_epochs': 1000}
INFO:Test Set AUC: 0.9171, Accuracy: 0.8502, Sensitivity: 0.8986

INFO:{'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2, 'learning_rate': 0.0001, 'num_epochs': 2000}
INFO:Test Set AUC: 0.9108, Accuracy: 0.8525, Sensitivity: 0.8675

INFO:{'hidden_size': 128, 'num_layers': 4, 'dropout': 0.2, 'learning_rate': 0.0001, 'num_epochs': 2000}
INFO:Test Set AUC: 0.8641, Accuracy: 0.8283, Sensitivity: 0.8696

INFO:{'hidden_size': 128, 'num_layers': 8, 'dropout': 0.2, 'learning_rate': 0.0001, 'num_epochs': 2000}
INFO:Test Set AUC: 0.9096, Accuracy: 0.8548, Sensitivity: 0.8613


INFO:{'hidden_size': 256, 'num_layers': 2, 'dropout': 0.2, 'learning_rate': 0.0001, 'num_epochs': 2000}
INFO:Test Set AUC: 0.9307, Accuracy: 0.8502, Sensitivity: 0.9275

INFO:{'hidden_size': 256, 'num_layers': 4, 'dropout': 0.2, 'learning_rate': 0.0001, 'num_epochs': 1000}
INFO:Test Set AUC: 0.9049, Accuracy: 0.8710, Sensitivity: 0.9006

INFO:{'hidden_size': 256, 'num_layers': 8, 'dropout': 0.2, 'learning_rate': 0.0001, 'num_epochs': 1000}
INFO:Test Set AUC: 0.8981, Accuracy: 0.8456, Sensitivity: 0.8530

INFO:{'hidden_size': 512, 'num_layers': 2, 'dropout': 0.2, 'learning_rate': 0.0001, 'num_epochs': 1000}
INFO:Test Set AUC: 0.9452, Accuracy: 0.8906, Sensitivity: 0.9296

INFO:{'hidden_size': 512, 'num_layers': 2, 'dropout': 0.2, 'learning_rate': 0.0001, 'num_epochs': 5000}
INFO:Test Set AUC: 0.9427, Accuracy: 0.8733, Sensitivity: 0.9213

INFO:{'hidden_size': 512, 'num_layers': 2, 'dropout': 0.2, 'learning_rate': 5e-05, 'num_epochs': 5000}
INFO:Test Set AUC: 0.9440, Accuracy: 0.8652, Sensitivity: 0.9234



INFO:{'hidden_size': 512, 'num_layers': 4, 'dropout': 0.2, 'learning_rate': 0.0001, 'num_epochs': 1000}
INFO:Test Set AUC: 0.9361, Accuracy: 0.8687, Sensitivity: 0.9275


INFO:{'hidden_size': 1024, 'num_layers': 2, 'dropout': 0.2, 'learning_rate': 0.0001, 'num_epochs': 1000}
INFO:Test Set AUC: 0.9453, Accuracy: 0.8790, Sensitivity: 0.9151


INFO:{'hidden_size': 1024, 'num_layers': 4, 'dropout': 0.2, 'learning_rate': 0.0001, 'num_epochs': 1000}
INFO:Test Set AUC: 0.9296, Accuracy: 0.8790, Sensitivity: 0.9275
    



INFO:{'hidden_size': 512, 'num_layers': 2, 'dropout': 0.2, 'learning_rate': 0.0001, 'num_epochs': 1000}
INFO:Test Set AUC: 0.9452, Accuracy: 0.8906, Sensitivity: 0.9296

INFO:{'hidden_size': 512, 'num_layers': 2, 'dropout': 0.2, 'learning_rate': 0.0001, 'num_epochs': 5000}
INFO:Test Set AUC: 0.9427, Accuracy: 0.8733, Sensitivity: 0.9213

INFO:{'hidden_size': 512, 'num_layers': 2, 'dropout': 0.2, 'learning_rate': 5e-05, 'num_epochs': 5000}
INFO:Test Set AUC: 0.9440, Accuracy: 0.8652, Sensitivity: 0.9234

INFO:{'hidden_size': 512, 'num_layers': 2, 'dropout': 0.4, 'learning_rate': 0.0001, 'num_epochs': 1000}
INFO:Test Set AUC: 0.9518, Accuracy: 0.8882, Sensitivity: 0.9234

INFO:{'hidden_size': 512, 'num_layers': 2, 'dropout': 0.6, 'learning_rate': 0.0001, 'num_epochs': 1000}
INFO:Test Set AUC: 0.9439, Accuracy: 0.8882, Sensitivity: 0.9172

INFO:{'hidden_size': 512, 'num_layers': 2, 'dropout': 0.5, 'learning_rate': 0.0001, 'num_epochs': 2000}
INFO:Test Set AUC: 0.9460, Accuracy: 0.8848, Sensitivity: 0.9151


```
