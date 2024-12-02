
# Assessing the Impact of Antihypertensive Combination on the Progression of Mild Cognitive Impairment to Alzheimer's Disease Using Electronic Health Records

**Transformer + LRSRegressor**
```
INFO:acc: 0.9225±0.0056
INFO:auc: 0.9531±0.0113
INFO:sen: 0.9330±0.0225
INFO:ate: 0.0004±0.0018
INFO:ate_low: -0.0041±0.0045
INFO:ate_up: 0.0049±0.0045
```

**Transformer + XGBRegressor**
```
INFO:acc: 0.9225±0.0056
INFO:auc: 0.9531±0.0113
INFO:sen: 0.9330±0.0225
INFO:ate: -0.0158±0.0152
INFO:ate_low: -0.0847±0.0091
INFO:ate_up: 0.0531±0.0258
```

**Transformer + RandomForestRegressor**
```
INFO:acc: 0.9225±0.0056
INFO:auc: 0.9531±0.0113
INFO:sen: 0.9330±0.0225
INFO:ate: 0.0000±0.0000
```

**LSTM + LRSRegressor**
```
INFO:acc: 0.8923±0.0074
INFO:auc: 0.9216±0.0119
INFO:sen: 0.9199±0.0291
INFO:ate: -0.0012±0.0035
INFO:ate_low: -0.0091±0.0043
INFO:ate_up: 0.0066±0.0036
```


**LSTM + XGBRegressor**
```
INFO:acc: 0.8923±0.0074
INFO:auc: 0.9216±0.0119
INFO:sen: 0.9199±0.0291
INFO:ate: -0.0011±0.0215
INFO:ate_low: -0.0922±0.0286
INFO:ate_up: 0.0899±0.0191
```

**LSTM + RandomForestRegressor**
```
INFO:acc: 0.8923±0.0074
INFO:auc: 0.9216±0.0119
INFO:sen: 0.9199±0.0291
INFO:ate: 0.0002±0.0009
```



