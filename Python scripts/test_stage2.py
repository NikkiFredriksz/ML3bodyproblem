import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
import torch.nn.functional as F
from torch.utils.data import random_split

# ==========================================
# CONFIGURATION
# ==========================================
# INPUTS
STORAGE_DB = "sqlite:///three_body_cascade_v20.db" 
TRAIN_FILE = "train3body.dat"
TEST_FILE = "data_for_stage2.csv" 

# OUTPUTS
SAVE_MODEL_FILE = "stage2_int_v22.pth"
SAVE_DATA_FILE = "data_for_stage3.csv"      

# SETTINGS (The "Landing" Tuning)
N_TRIALS = 0              # Set to 0 to use best saved params
EPOCHS_OPT = 10            # Fast epochs for trials
EPOCHS_INTERACTION = 140   # STOP before the spike (Peak Performance)
WEIGHT_INTERACTION = 2.0   # Relaxed from 3.0 to reduce False Positives
THRESH_INTERACTION = 0.70  # Stricter to filter out fake Exchanges

# ==========================================
# 1. PHYSICS ENGINE (v12 + Phase Features)
# ==========================================
class ThreeBodyPhysics:
    def __init__(self): 
        self.G = 4.302e-3

    def convert_batch_to_state(self, df, align=False):
        # 1. Unpack Variables
        m1 = df['m1'].values; m2 = df['m2'].values; m3 = df['m3'].values
        a = df['a_pc'].values; e = df['e'].values; b = df['b_pc'].values
        
        # Keep angles raw for rotation logic
        phi = np.where(np.abs(df['phi'].values)>2*np.pi, np.radians(df['phi'].values), df['phi'].values)
        theta = np.where(np.abs(df['theta'].values)>2*np.pi, np.radians(df['theta'].values), df['theta'].values)
        psi = np.where(np.abs(df['psi'].values)>2*np.pi, np.radians(df['psi'].values), df['psi'].values)
        f = df['f'].values; v_inf = df['v_km_s'].values; t_coal = df['t_coal_yr'].values
        
        M_bin = m1 + m2
        M_tot = m1 + m2 + m3
        
        # --- PHASE FEATURES ---
        r_peri_encounter = b 
        v_peri_encounter = np.sqrt(v_inf**2 + 2*self.G*M_bin/(r_peri_encounter+1e-9))
        v_avg = np.sqrt(v_inf * v_peri_encounter)
        t_approach = (50.0 * a) / (v_avg + 1e-9)

        mean_motion = np.sqrt(self.G * M_bin / (a**3 + 1e-9))
        M_encounter = f + mean_motion * t_approach
        
        feat_phase_sin = np.sin(M_encounter)
        feat_phase_cos = np.cos(M_encounter)
        
        # --- COORDINATE TRANSFORMS ---
        # 1. Calculate Magnitude (r_mag)
        r_mag = (a * (1 - e**2)) / (1 + e * np.cos(f))
        
        # 2. Calculate Angular Momentum (h) and Velocity Components
        term_h = self.G * M_bin * a * (1 - e**2)
        h_spec = np.sqrt(np.maximum(0.0, term_h))
        inv_h = np.zeros_like(h_spec); mask_h = h_spec > 0
        inv_h[mask_h] = 1.0 / h_spec[mask_h]
        
        vr = (self.G * M_bin * e * np.sin(f)) * inv_h
        vt = h_spec / (r_mag + 1e-9) # Fixed the broken line here
        
        # 3. Project to Plane
        c_f, s_f = np.cos(f), np.sin(f)
        r_rel_plane = np.stack([r_mag * c_f, r_mag * s_f, np.zeros_like(f)], axis=1)
        v_rel_plane = np.stack([vr * c_f - vt * s_f, vr * s_f + vt * c_f, np.zeros_like(f)], axis=1)
        
        # 4. Rotate to 3D Space
        z = np.zeros_like(phi); o = np.ones_like(phi)
        c, s = np.cos(phi), np.sin(phi); Rz_phi = np.stack([np.stack([c,-s,z],1), np.stack([s,c,z],1), np.stack([z,z,o],1)],1)
        c, s = np.cos(theta), np.sin(theta); Rx_theta = np.stack([np.stack([o,z,z],1), np.stack([z,c,-s],1), np.stack([z,s,c],1)],1)
        c, s = np.cos(psi), np.sin(psi); Rz_psi = np.stack([np.stack([c,-s,z],1), np.stack([s,c,z],1), np.stack([z,z,o],1)],1)
        R = Rz_phi @ Rx_theta @ Rz_psi
        
        r_rel = (R @ r_rel_plane[:,:,None]).squeeze(-1)
        v_rel = (R @ v_rel_plane[:,:,None]).squeeze(-1)
        
        r3 = np.stack([50*a, b, np.zeros_like(a)], axis=1)
        v3 = np.stack([-v_inf, np.zeros_like(v_inf), np.zeros_like(v_inf)], axis=1)

        # --- CALCULATE INCLINATION ---
        L_bin_vec = np.cross(r_rel, v_rel)
        L_outer_vec = np.cross(r3, v3)
        dot_L = np.sum(L_bin_vec * L_outer_vec, axis=1)
        norm_L = np.linalg.norm(L_bin_vec, axis=1) * np.linalg.norm(L_outer_vec, axis=1)
        cos_inclination = dot_L / (norm_L + 1e-9)
        
        # --- PHYSICS 1: ENERGIES ---
        E_bin = -self.G * m1 * m2 / (2 * a)
        E_inf = 0.5 * m3 * v_inf**2
        E_tot = E_bin + E_inf
        hardness_ratio = E_inf / (np.abs(E_bin) + 1e-9)

        # --- PHYSICS 2: MOMENTUM MAGNITUDES ---
        mu_bin = m1 * m2 / (M_bin + 1e-9)
        L_bin_mag = mu_bin * np.sqrt(self.G * M_bin * a * (1 - e**2) + 1e-9)
        mu_out = m3 * M_bin / (M_tot + 1e-9)
        L_inf_mag = mu_out * b * (v_inf + 1e-9)
        L_ratio = L_inf_mag / (L_bin_mag + 1e-9)

        # Helpers
        def lm(x): return np.sign(x)*np.log10(1+np.abs(x))
        r_peri = a * (1 - e)
        compactness = M_tot / (r_peri * (v_inf**2 + 1e-6) + 1e-9)
        
        # --- DERIVED GEOMETRY ---
        rm2 = (m2/M_bin)[:,None]; rm1 = (m1/M_bin)[:,None]
        r1 = -rm2*r_rel; r2 = rm1*r_rel
        d13 = np.linalg.norm(r1 - r3, axis=1)
        d23 = np.linalg.norm(r2 - r3, axis=1)
        
        # [NEW] Calculate Mass Ratio Proximity to 1 (The "Chaos Factor")
        q_prox = np.abs(1.0 - (m1/m2))
        
        # 5. Final Feature Assembly
        feat = [
            # Scales
            np.log10(m1)[:,None], np.log10(m2)[:,None], np.log10(m3)[:,None],
            np.log10(a)[:,None], 
            np.log10(np.maximum(1e-9, t_coal))[:,None],
            
            # Ratios
            (m1/m2)[:,None], (m2/m3)[:,None], (m3/m1)[:,None],
            
            # Energy
            lm(E_tot)[:,None],                 
            np.log10(hardness_ratio)[:,None],  
            
            # Momentum
            np.log10(L_ratio)[:,None],         
            
            # Geometry
            np.log10(r_peri + 1e-9)[:,None],
            np.log10((m3/M_bin)*(a/(b+1e-9))**3+1e-9)[:,None], 
            
            # Phases
            np.sin(f)[:,None], np.cos(f)[:,None],
            feat_phase_sin[:,None], feat_phase_cos[:,None],
            
            # Interaction
            lm(d13-d23)[:, None],
            cos_inclination[:, None],
            np.log10(compactness + 1e-9)[:,None],
            
            # [NEW FEATURE ADDED HERE]
            np.log10(q_prox + 1e-9)[:, None]
        ]
        return np.hstack(feat).astype(np.float32)

# ==========================================
# 2. DATASET (Interaction Mode)
# ==========================================
class CascadeDataset(Dataset):
    def __init__(self, filepath, physics_engine, mode='interaction', scaler=None, augment=False):
        if not os.path.exists(filepath): sys.exit(f"Error: {filepath} not found.")

        data = pd.read_csv(filepath, sep=r'\s+', engine='python')
        
        # Filter for Interaction Mode
        raw_outcomes = data['OUTCOME'].astype(int).values
        if mode == 'interaction':
            mask = raw_outcomes != 3
            data = data[mask].copy() # Filter DF first
            raw_outcomes = raw_outcomes[mask]
            self.y = (raw_outcomes > 0).astype(int)
            print(f"Dataset [Interaction]: {np.sum(self.y==1)} Pos (Exch) vs {np.sum(self.y==0)} Neg (Flyby)")
        
        # 1. Generate Original State
        self.X_orig = physics_engine.convert_batch_to_state(data)
        
        # 2. Generate Mirror State
        df_mirror = data.copy()
        df_mirror['m1'], df_mirror['m2'] = data['m2'], data['m1']
        df_mirror['psi'] += np.pi 
        self.X_mirror = physics_engine.convert_batch_to_state(df_mirror)
        
        # Note: Augmentation is removed because the Invariant Model handles it mathematically.
        
        if scaler:
            self.X_orig = scaler.transform(self.X_orig)
            self.X_mirror = scaler.transform(self.X_mirror)
            self.scaler = scaler
        else:
            self.scaler = StandardScaler()
            # Fit on both views to ensure symmetry in scaling
            combined = np.concatenate([self.X_orig, self.X_mirror], axis=0)
            self.scaler.fit(combined)
            self.X_orig = self.scaler.transform(self.X_orig)
            self.X_mirror = self.scaler.transform(self.X_mirror)
            
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): 
        return (torch.tensor(self.X_orig[idx]), 
                torch.tensor(self.X_mirror[idx]), 
                torch.tensor(self.y[idx], dtype=torch.long))

# ==========================================
# 3. MODEL
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout_rate)
        )
    def forward(self, x): return x + self.block(x)

class InvariantThreeBodyNet(nn.Module):
    """
    Wraps the base model to enforce physical Invariance.
    Unlike Stage 3 (which subtracts), Stage 2 ADDS because the class label 
    doesn't change when we swap the stars.
    Output = (Model(x) + Model(swap)) / 2
    """
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        
    def forward(self, x, x_mirror):
        # Pass both views
        out_orig = self.base(x)
        out_mirror = self.base(x_mirror)
        
        # Average them. This guarantees that Prediction(A,B) == Prediction(B,A)
        return (out_orig + out_mirror) / 2.0

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
# 4. OPTIMIZATION (OPTUNA) - NEW!
# ==========================================
def run_optimization(study_name, dataset, device):
    storage = optuna.storages.RDBStorage(url=STORAGE_DB)
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        print(f"Found existing study '{study_name}'.")
    except KeyError:
        print(f"Study '{study_name}' not found. Creating new one.")
        study = optuna.create_study(study_name=study_name, storage=storage, direction="minimize")

    if N_TRIALS == 0:
        if len(study.trials) > 0:
            print("Skipping optimization (Using BEST params from database).")
            return study.best_params
        else:
            print("No database history found and N_TRIALS=0. Using defaults.")
            return {'lr': 1e-3, 'hidden_dim': 512, 'num_layers': 4, 'dropout': 0.05, 'batch_size': 512}

    def objective(trial):
        # 1. Suggest Hyperparameters (UPDATED BATCH SIZE for GPU usage)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        drop = trial.suggest_float("dropout", 0.0, 0.4)
        bs = trial.suggest_categorical("batch_size", [2048, 4096, 8192]) # Increased from 512
        h_dim = trial.suggest_categorical("hidden_dim", [256, 512])
        n_layers = trial.suggest_int("num_layers", 2, 6)

        # 2. Fast Split (Train/Val)
        subset_size = int(0.25 * len(dataset)) 
        ds_subset, _ = random_split(dataset, [subset_size, len(dataset)-subset_size])
        
        t_size = int(0.8 * len(ds_subset))
        v_size = len(ds_subset) - t_size
        ds_t, ds_v = random_split(ds_subset, [t_size, v_size])
        
        # Optimization Loader (keep workers=0 here for safety during optuna)
        train_loader = DataLoader(ds_t, batch_size=bs, shuffle=True, num_workers=0, drop_last=True)
        val_loader = DataLoader(ds_v, batch_size=bs, shuffle=False, num_workers=0)

        # 3. Setup Model
        base_model = ThreeBodyResNet(dataset.X_orig.shape[1], 2, h_dim, n_layers, drop)
        model = InvariantThreeBodyNet(base_model).to(device)
        
        opt = optim.AdamW(model.parameters(), lr=lr)
        
        # [UPDATED WEIGHTS] Prioritize Flybys (Index 0) to fix 37% accuracy
        alpha = torch.tensor([1.5, 1.0]).to(device) 
    
        crit = FocalLoss(gamma=2.0, alpha=alpha)

        # 4. Training Loop
        for epoch in range(EPOCHS_OPT):
            model.train()
            for x, x_m, y in train_loader:
                x, x_m, y = x.to(device), x_m.to(device), y.to(device)
                opt.zero_grad()
                out = model(x, x_m)
                loss = crit(out, y.long())
                loss.backward()
                opt.step()
        
        # 5. Validation
        model.eval()
        val_loss = 0
        count = 0
        with torch.no_grad():
            for x, x_m, y in val_loader:
                x, x_m, y = x.to(device), x_m.to(device), y.to(device)
                out = model(x, x_m)
                loss = crit(out, y.long())
                val_loss += loss.item()
                count += 1
        return val_loss / count

    study.optimize(objective, n_trials=N_TRIALS)
    return study.best_params

# ==========================================
# 4. TRAIN FUNCTION
# ==========================================
def get_best_params_from_db(study_name, storage):
    print(f"Loading params from {study_name}...")
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        return study.best_params
    except:
        print("Could not load database. Using default params.")
        return {'lr': 1e-3, 'hidden_dim': 512, 'num_layers': 4, 'dropout': 0.05, 'batch_size': 512}

def train_stage2(dataset, params, device):
    print(f"\n--- Training Interaction (Weight={WEIGHT_INTERACTION}, Epochs={EPOCHS_INTERACTION}) ---")
    
    # Extract params 
    lr = params.get('lr', 1e-3)
    bs = params.get('batch_size', 4096) # Increased default
    h_dim = params.get('hidden_dim', 512)
    n_layers = params.get('num_layers', 4)
    drop = params.get('dropout', 0.05)

    counts = np.bincount(dataset.y)
    weights = 1. / (counts + 1e-6)
    sampler = WeightedRandomSampler(weights[dataset.y], len(dataset.y), replacement=True)
    
    # [GPU OPTIMIZATION]
    # Try num_workers=4. If Windows crashes, set to 0.
    # pin_memory=True speeds up transfer. persistent_workers=True keeps RAM ready.
    loader = DataLoader(dataset, batch_size=bs, sampler=sampler, 
                        num_workers=4, pin_memory=True, persistent_workers=True)
    
    # Enable Cudnn Benchmark
    torch.backends.cudnn.benchmark = True
    
    # Model Setup
    base_model = ThreeBodyResNet(dataset.X_orig.shape[1], 2, h_dim, n_layers, drop)
    model = InvariantThreeBodyNet(base_model).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    # [UPDATED WEIGHTS]
    alpha = torch.tensor([1.5, 1.0]).to(device) 

    criterion = FocalLoss(gamma=2.0, alpha=alpha)
    
    loss_hist = []
    lr_hist = []
    
    for epoch in range(EPOCHS_INTERACTION):
        model.train()
        total_loss = 0
        
        for x, x_m, y in loader:
            x, x_m, y = x.to(device), x_m.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x, x_m)
            loss = criterion(outputs, y.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        current_lr = optimizer.param_groups[0]['lr']
        loss_hist.append(total_loss/len(loader))
        lr_hist.append(current_lr)
        scheduler.step(epoch)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS_INTERACTION} | Loss: {loss_hist[-1]:.4f}")
            
    return model, loss_hist, lr_hist

# ==========================================
# 5. MAIN
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    physics = ThreeBodyPhysics()
    
    # 1. LOAD TRAINING DATA
    print("Initializing Training Data...")
    ds_train_fit = CascadeDataset(TRAIN_FILE, physics, mode='ionization', augment=False)
    # Note: augment=False because Invariant Model handles it
    ds_train = CascadeDataset(TRAIN_FILE, physics, mode='interaction', scaler=ds_train_fit.scaler, augment=False)
    
    # 2. RUN OPTIMIZATION
    print("\n--- Hyperparameter Optimization ---")
    # [CHANGED NAME] Forces fresh optimization for new weights
    best_params = run_optimization("opt_interaction_v23", ds_train, device) 
    print("Best Params:", best_params)

    # 3. TRAIN FINAL MODEL
    model, loss_hist, lr_hist = train_stage2(ds_train, best_params, device)
    
    print(f"Saving Model to {SAVE_MODEL_FILE}...")
    torch.save(model.state_dict(), SAVE_MODEL_FILE)

    # 4. EVALUATE ON FILTERED DATA
    if not os.path.exists(TEST_FILE):
        print(f"CRITICAL: '{TEST_FILE}' not found. Run Stage 1 script first.")
        sys.exit()
        
    print(f"\nEvaluating on {TEST_FILE} (Filtered from Stage 1)...")
    df_test = pd.read_csv(TEST_FILE, sep=r'\s+', engine='python')
    true_labels = (df_test['OUTCOME'].astype(int) > 0).astype(int).values
    
    # --- FIX: Use Vectorized Batch Conversion (Like Stage 3) ---
    # 1. Prepare Original View
    X_test_orig = physics.convert_batch_to_state(df_test)
    X_test_orig = torch.tensor(ds_train.scaler.transform(X_test_orig), dtype=torch.float32).to(device)
    
    # 2. Prepare Mirror View (Required for Invariant Model)
    df_test_mirror = df_test.copy()
    df_test_mirror['m1'], df_test_mirror['m2'] = df_test['m2'], df_test['m1']
    df_test_mirror['psi'] += np.pi
    X_test_mirror = physics.convert_batch_to_state(df_test_mirror)
    X_test_mirror = torch.tensor(ds_train.scaler.transform(X_test_mirror), dtype=torch.float32).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_orig, X_test_mirror)
        probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()

    # [NEW] CONDITIONAL THRESHOLDING (The Chaos Filter)
    # 1. Retrieve Mass Ratio
    m1_test = df_test['m1'].values
    m2_test = df_test['m2'].values
    mass_ratio = m1_test / (m2_test + 1e-9)
    
    # 2. Define Dynamic Thresholds
    base_thresh = 0.60    # Standard
    strict_thresh = 0.80  # Strict for chaotic zone
    
    # 3. Apply Conditional Logic
    dynamic_thresholds = np.where(
        (mass_ratio > 0.95) & (mass_ratio < 1.05), 
        strict_thresh, 
        base_thresh
    )
    
    # 4. Generate Predictions
    preds = (probs > dynamic_thresholds).astype(int)
    
    # 5. SAVE FILTERED DATA FOR STAGE 3
    indices_for_stage3 = [i for i, p in enumerate(preds) if p == 1]
    
    print(f"Total Stage 2 Inputs: {len(df_test)}")
    print(f"Classified as Fly-by: {np.sum(preds == 0)}")
    print(f"Passed to Stage 3 (Exchange): {len(indices_for_stage3)}")
    
    df_stage3 = df_test.iloc[indices_for_stage3].copy()
    df_stage3.to_csv(SAVE_DATA_FILE, sep=' ', index=False)
    print(f"Saved filtered data to '{SAVE_DATA_FILE}'")

    # 6. PLOTS
    cm = confusion_matrix(true_labels, preds)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    
    # Normalized Matrix (Percent)
    plt.figure(figsize=(8, 6))
    class_names = ['Fly-by', 'Exchange']
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Stage 2: Interaction Recall (Dynamic Thresh)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig("stage2_confusion_matrix.png")
    plt.show()

    # [NEW] UN-NORMALIZED CONFUSION MATRIX (RAW COUNTS)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Stage 2: Interaction Counts (Raw)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig("stage2_confusion_matrix_raw.png")
    plt.show()
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='blue')
    ax1.plot(range(1, EPOCHS_INTERACTION+1), loss_hist, color='blue', label='Loss')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Learning Rate', color='red')
    ax2.plot(range(1, EPOCHS_INTERACTION+1), lr_hist, color='red', linestyle='--', label='LR')
    ax2.tick_params(axis='y', labelcolor='red')
    plt.title("Stage 2 Training Dynamics")
    plt.tight_layout()
    plt.savefig("stage2_training_log.png")  # <--- ADDED THIS
    plt.show()
        
    # ==========================================
    # 5. ADVANCED DIAGNOSTICS & PLOTS
    # ==========================================
    print("\n--- Running Advanced Physics Diagnostics ---")
    
    # 1. PREPARE PHYSICS VARIABLES
    # We need to re-calculate these from the dataframe for plotting
    m1_test = df_test['m1'].values
    m2_test = df_test['m2'].values
    m3_test = df_test['m3'].values
    a_test = df_test['a_pc'].values
    v_inf_test = df_test['v_km_s'].values
    
    # Calculate Hardness Ratio (Kinetic / Binding Energy) - Key for Ionization
    G_const = 4.302e-3
    E_bin = -G_const * m1_test * m2_test / (2 * a_test)
    E_inf = 0.5 * m3_test * v_inf_test**2
    hardness_ratio_val = E_inf / (np.abs(E_bin) + 1e-9)
    
    # Mass Ratio
    mass_ratio_val = m1_test / (m2_test + 1e-9)
    
    # 2. CONFIDENCE HISTOGRAM
    plt.figure(figsize=(7,4))
    plt.hist(probs, bins=50, alpha=0.7, color='purple')
    plt.title("Distribution of Interaction Probabilities")
    plt.xlabel("Probability (0 = Ionization, 1 = Exchange)")
    plt.ylabel("Count")
    plt.axvline(THRESH_INTERACTION, color='red', linestyle='--', label=f'Threshold ({THRESH_INTERACTION})')
    plt.legend()
    plt.savefig("stage2_histogram.png")
    plt.show()

    # 3. PHYSICS ACCURACY PLOTS
    def plot_binned_accuracy(variable, var_name, y_true, y_pred, bins=10, log_scale=False):
        # Create bins
        if log_scale:
            bin_edges = np.logspace(np.log10(variable.min()+1e-9), np.log10(variable.max()), bins+1)
        else:
            bin_edges = np.linspace(variable.min(), variable.max(), bins+1)
            
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        accuracies = []
        counts = []
        
        for i in range(bins):
            mask = (variable >= bin_edges[i]) & (variable < bin_edges[i+1])
            if np.sum(mask) > 0:
                acc = np.mean(y_true[mask] == y_pred[mask])
                accuracies.append(acc)
                counts.append(np.sum(mask))
            else:
                accuracies.append(0)
                counts.append(0)
                
        plt.figure(figsize=(8, 4))
        plt.plot(bin_centers, accuracies, marker='o', linestyle='-', color='teal')
        plt.title(f"Accuracy vs. {var_name}")
        plt.xlabel(var_name)
        plt.ylabel("Accuracy")
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        if log_scale: plt.xscale('log')
        
        ax2 = plt.gca().twinx()
        ax2.bar(bin_centers, counts, width=np.diff(bin_edges), alpha=0.1, color='gray', align='edge')
        ax2.set_ylabel("Count")
        
        plt.savefig(f"stage2_acc_vs_{var_name.split()[0]}.png")
        plt.show()

    # Plot Accuracy vs Hardness (Log Scale is best for Energy ratios)
    plot_binned_accuracy(hardness_ratio_val, "Hardness Ratio (E_kin / E_bin)", true_labels, preds, log_scale=True)
    
    # Plot Accuracy vs Mass Ratio
    plot_binned_accuracy(mass_ratio_val, "Mass Ratio (m1/m2)", true_labels, preds, log_scale=False)

    # 4. FAILURE MAP (Hardness vs Mass Ratio)
    # This shows you exactly WHERE in parameter space the model fails
    plt.figure(figsize=(8, 6))
    correct_mask = (true_labels == preds)
    
    # We use Log Hardness for the Y-axis so it's readable
    log_hardness = np.log10(hardness_ratio_val + 1e-9)
    
    plt.scatter(mass_ratio_val[correct_mask], log_hardness[correct_mask], 
                c='green', s=5, alpha=0.1, label='Correct')
    plt.scatter(mass_ratio_val[~correct_mask], log_hardness[~correct_mask], 
                c='red', marker='x', s=20, alpha=0.6, label='Wrong')
    
    plt.title("Failure Map: Mass Ratio vs Hardness")
    plt.xlabel("Mass Ratio (m1/m2)")
    plt.ylabel("Log10 Hardness Ratio")
    plt.legend()
    plt.savefig("stage2_failure_map.png")
    plt.show()
    
    # 5. ROC CURVE
    fpr, tpr, thresholds = roc_curve(true_labels, probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig("stage2_roc.png")
    plt.show()
    
    print(f"Stage 2 Diagnostics Complete. ROC AUC: {roc_auc:.4f}")