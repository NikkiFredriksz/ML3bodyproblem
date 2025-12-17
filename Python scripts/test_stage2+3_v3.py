import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import argparse


# ==========================================
# CONFIGURATION
# ==========================================
STORAGE_DB = "sqlite:///three_body_cascade_v23.db" # Version 23

# OUTPUT MODELS
MODEL_S2_FILE = "stage2_int_v23.pth"
MODEL_S3_FILE = "stage3_exc_v23.pth"

# OPTIMIZATION SETTINGS
N_TRIALS = 15              # Trials per stage
EPOCHS_OPT = 10            # Fast epochs for finding params

# FINAL TRAINING SETTINGS
# Stage 2 (Imbalanced): Focal Loss, Strict Threshold
EPOCHS_S2 = 140
WEIGHT_S2 = 2.0   
THRESH_S2 = 0.70

# Stage 3 (Balanced): CrossEntropy, Augmentation, TTA
EPOCHS_S3 = 140   # Stopped at 140 (end of cycle) for optimal convergence
N_ROTATIONS = 2   # 4 Rotations -> 8 Views (4 Standard + 4 Swapped)

args = argparse.ArgumentParser()
args.add_argument("train_file", type=str, help="path naar train .dat")
args.add_argument("test_file", type=str, help="path naar test .dat")
args.add_argument("keyword", type=str, help="keyword")

TRAIN_FILE = args.train_file
TEST_FILE = args.test_file

# ==========================================
# 1. PHYSICS ENGINE (V23 - 41 FEATURES)
# ==========================================
class ThreeBodyPhysics:
    def __init__(self):
        self.G = 4.302e-3 

    def rotation_matrix(self, phi, theta, psi):
        Rz_phi = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
        Rx_theta = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
        Rz_psi = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
        return Rz_phi @ Rx_theta @ Rz_psi

    def convert_row_to_state(self, row, rotation_offset=0.0):
        # Extract Inputs
        m1, m2, m3 = row['m1'], row['m2'], row['m3']
        a, e, b = row['a_pc'], row['e'], row['b_pc']
        phi, theta, psi = row['phi'], row['theta'], row['psi']
        
        # Ensure radians
        if abs(phi) > 2 * np.pi or abs(theta) > 2 * np.pi or abs(psi) > 2 * np.pi:
            phi, theta, psi = np.radians(phi), np.radians(theta), np.radians(psi)

        # APPLY ROTATION OFFSET
        psi = (psi + rotation_offset) % (2 * np.pi)

        f, v_inf = row['f'], row['v_km_s'] 
        t_coal = row['t_coal_yr']
        
        # 1. Internal Binary State
        M_bin = m1 + m2
        r_mag = (a * (1 - e**2)) / (1 + e * np.cos(f))
        h_spec = np.sqrt(max(0.0, self.G * M_bin * a * (1 - e**2))) 
        if h_spec == 0: vr, vt = 0.0, 0.0
        else: vr, vt = (self.G * M_bin * e * np.sin(f)) / h_spec, h_spec / r_mag
             
        r_rel_plane = np.array([r_mag * np.cos(f), r_mag * np.sin(f), 0.0])
        v_rel_plane = np.array([vr * np.cos(f) - vt * np.sin(f), vr * np.sin(f) + vt * np.cos(f), 0.0])

        # Rotate to 3D inertial frame
        R = self.rotation_matrix(phi, theta, psi)
        r_rel = R @ r_rel_plane
        v_rel = R @ v_rel_plane

        # 2. Individual Bodies (CM Frame)
        r1, r2 = -(m2 / M_bin) * r_rel, (m1 / M_bin) * r_rel
        v1, v2 = -(m2 / M_bin) * v_rel, (m1 / M_bin) * v_rel

        r3, v3 = np.array([50 * a, b, 0.0]), np.array([-v_inf, 0.0, 0.0])
        M_tot = m1 + m2 + m3
        r_cm = (m1*r1 + m2*r2 + m3*r3) / M_tot
        v_cm = (m1*v1 + m2*v2 + m3*v3) / M_tot
        
        r1, r2, r3 = r1 - r_cm, r2 - r_cm, r3 - r_cm
        v1, v2, v3 = v1 - v_cm, v2 - v_cm, v3 - v_cm
        p1, p2, p3 = m1 * v1, m2 * v2, m3 * v3

        # 3. Standard Physics Features
        T = 0.5 * (m1 * np.linalg.norm(v1)**2 + m2 * np.linalg.norm(v2)**2 + m3 * np.linalg.norm(v3)**2)
        d12 = np.linalg.norm(r1 - r2)
        d13 = np.linalg.norm(r1 - r3)
        d23 = np.linalg.norm(r2 - r3)
        U = -self.G * ((m1*m2)/d12 + (m1*m3)/d13 + (m2*m3)/d23)
        
        def log_modulus(x): return np.sign(x) * np.log10(1 + np.abs(x))
        Total_Energy = T + U
        L_vec = np.cross(r1, p1) + np.cross(r2, p2) + np.cross(r3, p3)
        
        q12, q23, q31 = m1/m2, m2/m3, m3/m1
        mu = (m1 * m2) / (m1 + m2)
        L_mag = np.linalg.norm(L_vec) 
        virial = T / (abs(U) + 1e-9)
        r_peri = a * (1 - e)
        v_orb_proxy = np.sqrt(self.G * M_bin / a)
        impact_ratio = b / a
        vel_ratio = v_inf / (v_orb_proxy + 1e-9)

        # 4. NEW ADVANCED FEATURES
        # A. Tidal Parameter
        b_peri = max(1e-9, b * (1 - e))
        tidal_param = (m3 / M_bin) * np.power(a / b_peri, 3)

        # B. Angular Momentum Ratio
        mu_bin = (m1 * m2) / (m1 + m2)
        L_bin_vec = np.cross(r_rel, v_rel) * mu_bin
        L_bin_mag = np.linalg.norm(L_bin_vec)
        L_orb_mag_proxy = m3 * v_inf * b # Approx orbital angular momentum
        L_ratio = L_bin_mag / (L_orb_mag_proxy + 1e-9)

        # C. Spin-Orbit Misalignment
        # Cosine angle between L_bin and L_Total
        L_tot_mag = np.linalg.norm(L_vec)
        if L_bin_mag > 0 and L_tot_mag > 0:
            cos_spin_orbit = np.dot(L_bin_vec, L_vec) / (L_bin_mag * L_tot_mag)
        else:
            cos_spin_orbit = 0.0

        # RETURN 41 FEATURES
        return np.concatenate([
            r1, r2, r3, p1, p2, p3, r_rel, v_rel,          # 9+9+6 = 24
            [log_modulus(Total_Energy)], [log_modulus(val) for val in L_vec], # 1+3 = 4
            [np.log10(max(1e-9, t_coal))], [np.log10(mu)], [q12, q23, q31],   # 1+1+3 = 5              
            [np.log10(L_mag + 1e-9), np.log10(virial + 1e-9), np.log10(r_peri + 1e-9)], # 3
            [np.log10(impact_ratio + 1e-9), np.log10(vel_ratio + 1e-9)],      # 2
            [np.log10(tidal_param + 1e-9), np.log10(L_ratio + 1e-9), cos_spin_orbit] # 3
        ]) # Total = 41

# ==========================================
# 2. DATASET
# ==========================================
class CascadeDataset(Dataset):
    def __init__(self, filepath, physics_engine, mode='interaction', scaler=None, augment=False, n_rotations=0):
        if not os.path.exists(filepath):
            print(f"Error: {filepath} not found.")
            sys.exit()

        data = pd.read_csv(filepath, sep=r'\s+', engine='python')
        self.feature_cols = ['m1', 'm2', 'm3', 'a_pc', 'e', 'b_pc', 'phi', 'theta', 'psi', 'f', 'v_km_s', 'Ecc_Anomaly', 't_coal_yr']
        raw_outcomes = data['OUTCOME'].astype(int).values

        # MODE SELECTION
        if mode == 'ionization': # For scaler fit
            self.y = (raw_outcomes == 3).astype(int)
            data_subset = data.copy()
            
        elif mode == 'interaction': # Stage 2
            mask = raw_outcomes != 3
            data_subset = data[mask].copy()
            raw_outcomes = raw_outcomes[mask]
            self.y = (raw_outcomes > 0).astype(int) # 1=Exchange, 0=Flyby
            
        elif mode == 'exchange': # Stage 3
            mask = (raw_outcomes == 1) | (raw_outcomes == 2)
            data_subset = data[mask].copy()
            raw_outcomes = raw_outcomes[mask]
            self.y = (raw_outcomes == 2).astype(int) # 1=Exch1-3(Outcome 2), 0=Exch2-3(Outcome 1)

        # GENERATE FEATURES
        X_list = []
        y_list = []
        
        # Base Data
        base_states = [physics_engine.convert_row_to_state(row, rotation_offset=0.0) for _, row in data_subset[self.feature_cols].iterrows()]
        X_list.extend(base_states)
        y_list.extend(self.y)

        # AUGMENTATION
        if augment:
            # Angles
            if n_rotations > 0:
                angles = np.linspace(0, 2*np.pi, n_rotations + 1)[:-1]
                if n_rotations > 1: angles = angles[1:] # Skip 0
            else:
                angles = []

            # 1. Rotations
            for angle in angles:
                rot_states = [physics_engine.convert_row_to_state(row, rotation_offset=angle) for _, row in data_subset[self.feature_cols].iterrows()]
                X_list.extend(rot_states)
                y_list.extend(self.y)
            
            # 2. Swaps
            swapped = data_subset[self.feature_cols].copy()
            swapped['m1'], swapped['m2'] = data_subset['m2'], data_subset['m1']
            
            if mode == 'exchange': y_swapped = 1 - self.y # Flip label
            else: y_swapped = self.y # Keep label (Exch is Exch)
            
            # Swap Base
            swap_states = [physics_engine.convert_row_to_state(row, rotation_offset=0.0) for _, row in swapped.iterrows()]
            X_list.extend(swap_states)
            y_list.extend(y_swapped)
            
            # Swap Rotations
            for angle in angles:
                rot_swap_states = [physics_engine.convert_row_to_state(row, rotation_offset=angle) for _, row in swapped.iterrows()]
                X_list.extend(rot_swap_states)
                y_list.extend(y_swapped)

        self.X = np.array(X_list, dtype=np.float32)
        self.y = np.array(y_list, dtype=np.int64)

        if scaler:
            self.X = scaler.transform(self.X)
            self.scaler = scaler
        else:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X)
            
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return torch.tensor(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)

# ==========================================
# 3. MODEL COMPONENTS
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout_rate)
        )
    def forward(self, x): return x + self.block(x)

class ThreeBodyResNet(nn.Module):
    def __init__(self, input_dim, output_dim=2, hidden_dim=512, num_layers=4, dropout_rate=0.05):
        super(ThreeBodyResNet, self).__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU())
        layers = []
        for _ in range(num_layers):
            layers.append(ResidualBlock(hidden_dim, dropout_rate))
        self.res_blocks = nn.Sequential(*layers)
        self.output_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, output_dim))
    def forward(self, x): return self.output_head(self.res_blocks(self.input_layer(x)))

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        return focal_loss.sum()

# ==========================================
# 4. OPTIMIZATION & TRAINING
# ==========================================
def run_optimization(study_name, dataset, device, mode='interaction'):
    print(f"\n[Optuna] Optimizing {study_name}...")
    
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512]) 
        num_layers = trial.suggest_int("num_layers", 2, 6) 
        dropout = trial.suggest_float("dropout", 0.0, 0.2)
        batch_size = trial.suggest_categorical("batch_size", [256, 512])
        
        train_size = int(0.2 * len(dataset))
        test_size = int(0.05 * len(dataset))
        rest = len(dataset) - train_size - test_size
        ds_train, ds_test, _ = random_split(dataset, [train_size, test_size, rest])
        
        loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
        loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
        
        model = ThreeBodyResNet(dataset.X.shape[1], 2, hidden_dim, num_layers, dropout).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        
        if mode == 'interaction': criterion = FocalLoss(gamma=2.0)
        else: criterion = nn.CrossEntropyLoss()
             
        for epoch in range(EPOCHS_OPT):
            model.train()
            for inputs, labels in loader_train:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, labels in loader_test:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            acc = correct / total
            trial.report(acc, epoch)
            if trial.should_prune(): raise optuna.TrialPruned()
            
        return acc

    study = optuna.create_study(study_name=study_name, storage=STORAGE_DB, direction="maximize", load_if_exists=True)
    if len(study.trials) < N_TRIALS:
        study.optimize(objective, n_trials=N_TRIALS - len(study.trials))
    
    print(f"  -> Best Params: {study.best_params}")
    return study.best_params

def train_final_model(name, dataset, params, device, epochs, mode='interaction', weight=None):
    print(f"\n--- Final Training {name} ({epochs} Epochs) ---")
    
    counts = np.bincount(dataset.y)
    weights_sample = 1. / (counts + 1e-6)
    sampler = WeightedRandomSampler(weights_sample[dataset.y], len(dataset.y), replacement=True)
    
    loader = DataLoader(dataset, batch_size=params['batch_size'], sampler=sampler)
    model = ThreeBodyResNet(dataset.X.shape[1], 2, params['hidden_dim'], params['num_layers'], params['dropout']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=params['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    if mode == 'interaction':
        print(f"  -> Using Focal Loss (Alpha=1:{weight})")
        alpha = torch.tensor([1.0, weight]).to(device)
        criterion = FocalLoss(gamma=2.0, alpha=alpha)
    else:
        print("  -> Using CrossEntropyLoss")
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        
    loss_hist, lr_hist = [], []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step(epoch)
        loss_hist.append(total_loss/len(loader))
        lr_hist.append(optimizer.param_groups[0]['lr'])
        
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss_hist[-1]:.4f}")
            
    return model, loss_hist, lr_hist

# ==========================================
# 5. MAIN
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    physics = ThreeBodyPhysics()
    
    # 0. INITIAL SCALER FIT
    print("Fitting Scaler...")
    ds_fit = CascadeDataset(TRAIN_FILE, physics, mode='ionization', augment=False)
    
    # --- STAGE 2: INTERACTION ---
    print("\n=== STAGE 2 SETUP ===")
    ds_s2_opt = CascadeDataset(TRAIN_FILE, physics, mode='interaction', scaler=ds_fit.scaler, augment=False)
    params_s2 = run_optimization("opt_s2_v23", ds_s2_opt, device, mode='interaction')
    
    # Train S2 (Augmentation + Focal Loss)
    ds_s2_final = CascadeDataset(TRAIN_FILE, physics, mode='interaction', scaler=ds_fit.scaler, augment=True)
    model_s2, loss_s2, lr_s2 = train_final_model("Stage 2", ds_s2_final, params_s2, device, EPOCHS_S2, mode='interaction', weight=WEIGHT_S2)
    torch.save(model_s2.state_dict(), MODEL_S2_FILE)
    
    # --- STAGE 3: EXCHANGE ---
    print("\n=== STAGE 3 SETUP ===")
    ds_s3_opt = CascadeDataset(TRAIN_FILE, physics, mode='exchange', scaler=ds_fit.scaler, augment=False)
    params_s3 = run_optimization("opt_s3_v23", ds_s3_opt, device, mode='exchange')
    
    # Train S3 (Rotations + CrossEntropy)
    ds_s3_final = CascadeDataset(TRAIN_FILE, physics, mode='exchange', scaler=ds_fit.scaler, augment=True, n_rotations=N_ROTATIONS)
    model_s3, loss_s3, lr_s3 = train_final_model("Stage 3", ds_s3_final, params_s3, device, EPOCHS_S3, mode='exchange')
    torch.save(model_s3.state_dict(), MODEL_S3_FILE)

    # --- FINAL EVALUATION ---
    print("\n=== FINAL SEQUENTIAL EVALUATION ===")
    df_test = pd.read_csv(TEST_FILE, sep=r'\s+', engine='python')
    df_test = df_test[df_test['OUTCOME'] != 3].copy() 
    true_labels = df_test['OUTCOME'].astype(int).values
    
    X_raw = np.array([physics.convert_row_to_state(row) for _, row in df_test[ds_s2_final.feature_cols].iterrows()])
    X_test = torch.tensor(ds_fit.scaler.transform(X_raw), dtype=torch.float32).to(device)
    
    model_s2.eval()
    model_s3.eval()
    final_preds = []
    
    # TTA Setup
    angles = np.linspace(0, 2*np.pi, N_ROTATIONS + 1)[:-1]

    with torch.no_grad():
        # S2 PREDICTION
        probs_s2 = F.softmax(model_s2(X_test), dim=1)[:, 1].cpu().numpy()
        preds_s2 = (probs_s2 > THRESH_S2).astype(int)
        
        # S3 PREDICTION (With TTA Averaging)
        s3_probs_sum = np.zeros(len(df_test))
        
        # Average over rotations
        for angle in angles:
            # Standard
            states = [physics.convert_row_to_state(row, rotation_offset=angle) for _, row in df_test[ds_s3_final.feature_cols].iterrows()]
            X_rot = torch.tensor(ds_fit.scaler.transform(states), dtype=torch.float32).to(device)
            s3_probs_sum += F.softmax(model_s3(X_rot), dim=1)[:, 1].cpu().numpy()
            
            # Swapped
            df_swap = df_test.copy()
            df_swap['m1'], df_swap['m2'] = df_swap['m2'], df_swap['m1']
            states_swap = [physics.convert_row_to_state(row, rotation_offset=angle) for _, row in df_swap[ds_s3_final.feature_cols].iterrows()]
            X_swap = torch.tensor(ds_fit.scaler.transform(states_swap), dtype=torch.float32).to(device)
            probs_swap = F.softmax(model_s3(X_swap), dim=1)[:, 1].cpu().numpy()
            s3_probs_sum += (1.0 - probs_swap)
            
        s3_final_probs = s3_probs_sum / (2 * len(angles))
        preds_s3 = (s3_final_probs > 0.5).astype(int)
        
        for i in range(len(true_labels)):
            if preds_s2[i] == 0:
                final_preds.append(0) # Flyby
            else:
                if preds_s3[i] == 0: final_preds.append(1) # Exch 2-3
                else: final_preds.append(2) # Exch 1-3
    
    cm = confusion_matrix(true_labels, final_preds)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=['Flyby', 'Exch 2-3', 'Exch 1-3'], yticklabels=['Flyby', 'Exch 2-3', 'Exch 1-3'])
    plt.title('Final Pipeline (S2 Focal + S3 Rotations + 41 Features)')
    plt.show()
    
    plt.figure(figsize=(10, 5))
    plt.plot(loss_s2, label='Stage 2 Loss')
    plt.plot(loss_s3, label='Stage 3 Loss')
    plt.legend()
    plt.title("Training Dynamics")
    plt.show()