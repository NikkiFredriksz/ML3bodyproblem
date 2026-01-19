# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 09:33:14 2026

@author: Lenovo T14 Gen 2
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 16:25:18 2025

@author: Lenovo T14 Gen 2
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 15:31:14 2025

@author: Lenovo T14 Gen 2
"""
import torch
import torch.nn as nn
import torch.optim as optim
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
# import random
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.metrics import ConfusionMatrixDisplay
import optuna
import argparse
import os
#%% 
#inputdata
n_trials=1
n_jobs=-1
epochstry=600
epochtest=600
hidden_dims=128
# # --- Traindata inlezen ---
# file = r"C:\Users\Lenovo T14 Gen 2\ML3body\ML3bodyproblem\GCdata\train3body.dat"
# df = pd.read_csv(file, delim_whitespace=True, header=0)
# # --- Testdata inlezen ---
# file_test = r"C:\Users\Lenovo T14 Gen 2\ML3body\ML3bodyproblem\GCdata\test3body.dat"
# df_test = pd.read_csv(file_test, delim_whitespace=True, header=0)

parser = argparse.ArgumentParser()
parser.add_argument("train_file", type=str, help="path naar train .dat")
parser.add_argument("test_file",  type=str, help="path naar test .dat")
#parser.add_argument("Keyword", type=str, help="The model keyword")
args = parser.parse_args()

# ----------------------
# Data inlezen
# ----------------------
df = pd.read_csv(args.train_file, delim_whitespace=True, header=0)
df_test = pd.read_csv(args.test_file, delim_whitespace=True, header=0)
#%% periodicity en E's
angle_columns = ["phi", "theta", "psi","f"]
for col in angle_columns:
    df[col + "_sin"] = np.sin(df[col])
    df[col + "_cos"] = np.cos(df[col])
    df_test[col + "_sin"] = np.sin(df_test[col])
    df_test[col + "_cos"] = np.cos(df_test[col])
df = df.drop(columns=angle_columns)
df_test=df_test.drop(columns=angle_columns)

df["E_bin"]=df["m1"]*df["m2"]/df["a_pc"]
df["E_kin3"]=df["m3"]*df["v_km_s"]**2
df["E_pot"]=(df["m1"]+df["m2"])*df["m3"]/df["b_pc"]
df["mratio1"]=df["m1"]/df["m2"]
df["mratio2"]=(df["m1"]+df["m2"])/df["m3"]
df["tbinary"]=df["a_pc"]**1.5*(df["m1"]+df["m2"])**(-0.5)
df["t_ratio"]=df["tbinary"]/df["t_coal_yr"]

df_test["E_bin"]=df_test["m1"]*df_test["m2"]/df_test["a_pc"]
df_test["E_kin3"]=df_test["m3"]*df_test["v_km_s"]**2
df_test["E_pot"]=(df_test["m1"]+df_test["m2"])*df_test["m3"]/df_test["b_pc"]
df_test["mratio1"]=df_test["m1"]/df_test["m2"]
df_test["mratio2"]=(df_test["m1"]+df_test["m2"])/df_test["m3"]
df_test["tbinary"]=df_test["a_pc"]**1.5*(df_test["m1"]+df_test["m2"])**(-0.5)
df_test["t_ratio"]=df_test["tbinary"]/df_test["t_coal_yr"]
featuresold = [
    "m1", "m2", "m3", "a_pc", "e", "b_pc",
    "phi_sin", "phi_cos",
    "theta_sin", "theta_cos",
    "psi_sin", "psi_cos",
    "f_sin", "f_cos", "v_km_s", "Ecc_Anomaly", "t_coal_yr"
]

features = [
    "m1", "m2", "m3", "a_pc", "e", "b_pc",
    "phi_sin", "phi_cos",
    "theta_sin", "theta_cos",
    "psi_sin", "psi_cos",
    "f_sin", "f_cos", "v_km_s", "Ecc_Anomaly", "t_coal_yr","E_bin","E_kin3","E_pot","mratio1",
    "mratio2","tbinary", "t_ratio"
]
#%% optuna voor 0,1,2

X_train = df[features].values
y_train = df["OUTCOME"].values

trainmask = (y_train != 3)


X_train=X_train[trainmask]
y_train=y_train[trainmask]

X_test = df_test[features].values
y_test = df_test["OUTCOME"].values
testmask = (y_test != 3)
X_test=X_test[testmask]
y_test=y_test[testmask]


# Schalen
scaler = StandardScaler()
#scaler=MinMaxScaler()
#scaler=RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# numpy → torch
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test  = torch.tensor(y_test, dtype=torch.long)

# --- Simpel NN, deze kan worden uitgebreid, maar het lijkt niet echt uit te maken ---
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
    
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
    
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
    
        nn.Linear(hidden_dim, 3)
        )
    def forward(self, x):
        return self.net(x)
    
model = SimpleNN(input_dim=X_train.shape[1], hidden_dim=hidden_dims)

weights = np.array([1.6937268170024657,  0.7856030253257161, 1.4252399618088845]) #van deep

weights = weights / weights.sum()           # optioneel normaliseren
weights = torch.tensor(weights, dtype=torch.float32)


learningrates=[20e-3]
epochs=np.arange(1,epochtest,1)
lossperepoc=np.zeros((len(epochs),len(learningrates)))
vallossperepoch = np.zeros((len(epochs), len(learningrates)))
accuracy=np.zeros(len(learningrates))
i=0
j=0
# --- Initialize lists/arrays om metrics bij te houden ---
accuracies = []
macro_f1s = []
recalls = {i: [] for i in range(3)}  # recall per klasse
precisions = {i: [] for i in range(3)}  # precision per klasse

for lr_idx, lr in enumerate(learningrates):
    model = SimpleNN(input_dim=X_train.shape[1], hidden_dim=hidden_dims)  # <-- nieuw model
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=weights)
    # criterion = nn.CrossEntropyLoss()
    for epoch_idx, epoch in enumerate(epochs):
    
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        lossperepoc[epoch_idx, lr_idx] = loss.item()

        optimizer.step()
        # --- validation loss NA training, zonder gradient ---
        with torch.no_grad():
            val_logits = model(X_test)
            val_loss = criterion(val_logits, y_test)
        
            vallossperepoch[epoch_idx, lr_idx] = val_loss.item()
        # --- Metrics per epoch (optioneel elke N epochs)
        if epoch_idx % 50 == 0:  # bijv elke 50e
            print(epoch_idx)
            with torch.no_grad():
                preds = torch.argmax(model(X_test), dim=1)
                y_true = y_test.numpy()
                y_pred = preds.numpy()
                
                acc = (preds == y_test).float().mean().item()
                f1 = f1_score(y_true, y_pred, average='macro')
                
                accuracies.append((lr, epoch, acc))
                macro_f1s.append((lr, epoch, f1))
                for i in range(3):
                    recalls[i].append((lr, epoch, recall_score(y_true, y_pred, labels=[i], average='macro')))
                    precisions[i].append((lr, epoch, precision_score(y_true, y_pred, labels=[i], average='macro')))
        

#%%
study_name=["long run"]
os.makedirs("results", exist_ok=True)
plt.figure()
with torch.no_grad():
    preds = torch.argmax(model(X_test), dim=1).numpy()
    y_true = y_test.numpy()

cm = confusion_matrix(y_true, preds)

disp = ConfusionMatrixDisplay.from_predictions(
    y_true, preds,
    display_labels=[0,1,2],
    cmap=plt.cm.Blues,
    normalize='true'     # <–– dit is de key
    )
disp.plot(cmap=plt.cm.Blues)
plt.title(f'Confusion Matrix (lr={lr}) ' +study_name)
plt.savefig("./results/confusionmatrix "+study_name)
plt.figure(figsize=(10,6))
for lr in learningrates:
    epochs_plot = [e for l,e,_ in accuracies if l==lr]
    acc_plot = [a for l,e,a in accuracies if l==lr]
    plt.plot(epochs_plot, acc_plot, label=f'accuracy lr={lr}')
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.title("NN Accuracy "+study_name)
plt.savefig("./results/accuracy "+study_name)


plt.figure(figsize=(10,6))
for lr_idx, lr in enumerate(learningrates):
    plt.plot(epochs, lossperepoc[:, lr_idx], label=f'loss lr={lr}')
    plt.plot(epochs, vallossperepoch[:, lr_idx], label=f'validation loss lr={lr}')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Loss per epoch "+study_name)
plt.legend()
plt.savefig("./results/loss "+study_name)

plt.figure(figsize=(10,6))
for i in range(3):
    epochs_plot = [e for l,e,_ in recalls[i] if l==learningrates[0]]  # bv 1e lr
    recall_plot = [r for l,e,r in recalls[i] if l==learningrates[0]]
    plt.plot(epochs_plot, recall_plot, label=f"recall class {i}")
plt.xlabel("epoch")
plt.ylabel("recall")
plt.title(f"Recall per class (lr={learningrates[0]}) "+study_name)
plt.legend()

plt.savefig("./results/recall "+study_name)
