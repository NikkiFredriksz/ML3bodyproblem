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
from sklearn.metrics import roc_curve, auc

# ==========================================
# CONFIGURATION
# ==========================================
torch.set_float32_matmul_precision('medium') 

STORAGE_DB = "sqlite:///three_body_stage3_v10.db"
TRAIN_FILE = "train3body.dat"
TEST_FILE = "test3body.dat"
MODEL_S3_PREFIX = "stage3_siamese_v5" 

# HYPERPARAMETERS
N_TRIALS = 100          # Set to 0 to use best existing params without re-optimizing
EPOCHS_OPT = 12         # Epochs for optimization trials

# STAGE 3 SETTINGS (ENSEMBLE)
N_MODELS_S3 = 10         # Number of brains in the committee
EPOCHS_S3 = 140         # Deep training for final models
ALIGN_TO_BINARY = True 

# STRATEGY
AUGMENT_TRAIN = False   # Not needed for Siamese (it sees both views automatically)
AUGMENT_TEST = False     # Evaluates symmetry
NUM_WORKERS = 0         # Set to 0 for Windows, increase for Linux


# ==========================================
# 1. PHYSICS ENGINE (v12 + Phase Features)
# ==========================================
class ThreeBodyPhysics:
    def __init__(self): self.G = 4.302e-3 
    
    def convert_batch_to_state(self, df, align=False):
        # 1. Unpack Variables
        m1 = df['m1'].values; m2 = df['m2'].values; m3 = df['m3'].values
        a = df['a_pc'].values; e = df['e'].values; b = df['b_pc'].values
        
        # Keep angles raw for rotation logic first
        phi = np.where(np.abs(df['phi'].values)>2*np.pi, np.radians(df['phi'].values), df['phi'].values)
        theta = np.where(np.abs(df['theta'].values)>2*np.pi, np.radians(df['theta'].values), df['theta'].values)
        psi = np.where(np.abs(df['psi'].values)>2*np.pi, np.radians(df['psi'].values), df['psi'].values)
        f = df['f'].values; v_inf = df['v_km_s'].values; t_coal = df['t_coal_yr'].values
        
        M_bin = m1 + m2
        r_mag = (a * (1 - e**2)) / (1 + e * np.cos(f))

        # --- UPGRADE: DEFINITIONS FOR SIN/COS ANGLES ---
        sin_phi, cos_phi = np.sin(phi), np.cos(phi)
        sin_theta, cos_theta = np.sin(theta), np.cos(theta)
        sin_psi, cos_psi = np.sin(psi), np.cos(psi)
        sin_f, cos_f = np.sin(f), np.cos(f)
        
        # --- UPGRADE: Smart Phase Features ---
        r_peri_encounter = b 
        v_peri_encounter = np.sqrt(v_inf**2 + 2*self.G*M_bin/(r_peri_encounter+1e-9))
        v_avg = np.sqrt(v_inf * v_peri_encounter)
        t_approach = (50.0 * a) / (v_avg + 1e-9)

        mean_motion = np.sqrt(self.G * M_bin / (a**3 + 1e-9))
        M_encounter = f + mean_motion * t_approach
        
        feat_phase_sin = np.sin(M_encounter)
        feat_phase_cos = np.cos(M_encounter)
        
        # 3. Coordinate Transformations
        term_h = self.G * M_bin * a * (1 - e**2)
        h_spec = np.sqrt(np.maximum(0.0, term_h))
        inv_h = np.zeros_like(h_spec); mask_h = h_spec > 0
        inv_h[mask_h] = 1.0 / h_spec[mask_h]
        vr = (self.G * M_bin * e * np.sin(f)) * inv_h
        vt = h_spec / r_mag
        c_f, s_f = np.cos(f), np.sin(f)
        r_rel_plane = np.stack([r_mag * c_f, r_mag * s_f, np.zeros_like(f)], axis=1)
        v_rel_plane = np.stack([vr * c_f - vt * s_f, vr * s_f + vt * c_f, np.zeros_like(f)], axis=1)
        
        z = np.zeros_like(phi); o = np.ones_like(phi)
        c, s = np.cos(phi), np.sin(phi); Rz_phi = np.stack([np.stack([c,-s,z],1), np.stack([s,c,z],1), np.stack([z,z,o],1)],1)
        c, s = np.cos(theta), np.sin(theta); Rx_theta = np.stack([np.stack([o,z,z],1), np.stack([z,c,-s],1), np.stack([z,s,c],1)],1)
        c, s = np.cos(psi), np.sin(psi); Rz_psi = np.stack([np.stack([c,-s,z],1), np.stack([s,c,z],1), np.stack([z,z,o],1)],1)
        R = Rz_phi @ Rx_theta @ Rz_psi
        r_rel = (R @ r_rel_plane[:,:,None]).squeeze(-1)
        v_rel = (R @ v_rel_plane[:,:,None]).squeeze(-1)
        
        rm2 = (m2/M_bin)[:,None]; rm1 = (m1/M_bin)[:,None]
        r1 = -rm2*r_rel; r2 = rm1*r_rel; v1 = -rm2*v_rel; v2 = rm1*v_rel
        r3 = np.stack([50*a, b, np.zeros_like(a)], axis=1)
        v3 = np.stack([-v_inf, np.zeros_like(v_inf), np.zeros_like(v_inf)], axis=1)
        
        M_tot = (m1+m2+m3)[:,None]
        r_cm = (m1[:,None]*r1 + m2[:,None]*r2 + m3[:,None]*r3)/M_tot
        v_cm = (m1[:,None]*v1 + m2[:,None]*v2 + m3[:,None]*v3)/M_tot
        r1-=r_cm; r2-=r_cm; r3-=r_cm; v1-=v_cm; v2-=v_cm; v3-=v_cm
        
        if align:
            aa = -psi; ca, sa = np.cos(aa), np.sin(aa)
            Ra = np.stack([np.stack([ca,-sa,z],1), np.stack([sa,ca,z],1), np.stack([z,z,o],1)],1)
            r1=(Ra@r1[:,:,None]).squeeze(-1); r2=(Ra@r2[:,:,None]).squeeze(-1); r3=(Ra@r3[:,:,None]).squeeze(-1)
            v1=(Ra@v1[:,:,None]).squeeze(-1); v2=(Ra@v2[:,:,None]).squeeze(-1); v3=(Ra@v3[:,:,None]).squeeze(-1)
            r_rel=(Ra@r_rel[:,:,None]).squeeze(-1); v_rel=(Ra@v_rel[:,:,None]).squeeze(-1)

        p1=m1[:,None]*v1; p2=m2[:,None]*v2; p3=m3[:,None]*v3
        
        d13=np.linalg.norm(r1-r3, axis=1); d23=np.linalg.norm(r2-r3, axis=1)
        E13=-self.G*m1*m3/d13; E23=-self.G*m2*m3/d23
        diff_d = d13 - d23

        L13 = np.cross(r1-r3, v1-v3); magL13 = np.linalg.norm(L13, axis=1)
        L23 = np.cross(r2-r3, v2-v3); magL23 = np.linalg.norm(L23, axis=1)
        diff_L = magL13 - magL23
        
        L_bin_vec = np.cross(r_rel, v_rel)
        L_outer_vec = np.cross(r3, v3)
        dot_L = np.sum(L_bin_vec * L_outer_vec, axis=1)
        norm_L = np.linalg.norm(L_bin_vec, axis=1) * np.linalg.norm(L_outer_vec, axis=1)
        cos_inclination = dot_L / (norm_L + 1e-9)

        def lm(x): return np.sign(x)*np.log10(1+np.abs(x))
        r_peri = a * (1 - e)
        compactness = M_tot.squeeze() / (r_peri * (v_inf**2 + 1e-6) + 1e-9)

        # 5. Final Feature Assembly
        feat = [
            r1, r2, r3, p1, p2, p3, r_rel, v_rel, 
            np.log10(np.maximum(1e-9, t_coal))[:,None], 
            np.log10(m1)[:,None], np.log10(m2)[:,None], np.log10(m3)[:,None], 
            np.log10(a)[:,None], 
            (m1/m2)[:,None], (m2/m3)[:,None], (m3/m1)[:,None], 
            np.log10(r_peri + 1e-9)[:,None],
            np.log10((m3/M_bin)*(a/(b+1e-9))**3+1e-9)[:,None], 
            lm(E13-E23)[:,None], lm(E13/(E23+1e-9))[:,None], 
            diff_d[:, None], lm(diff_L)[:, None],
            cos_inclination[:, None], np.log10(compactness + 1e-9)[:,None],
            sin_phi[:,None], cos_phi[:,None],
            sin_theta[:,None], cos_theta[:,None],
            sin_psi[:,None], cos_psi[:,None],
            sin_f[:,None], cos_f[:,None],
            feat_phase_sin[:,None], feat_phase_cos[:,None]
        ]
        return np.hstack(feat).astype(np.float32)

# ==========================================
# 2. DATASET
# ==========================================
class CascadeSiameseDataset(Dataset):
    def __init__(self, filepath, physics_engine, scaler=None, align=False, augment=False):
        if not os.path.exists(filepath): sys.exit(f"File {filepath} not found.")
        data = pd.read_csv(filepath, sep=r'\s+', engine='python')
        
        # ... (Standard outcomes filter) ...
        raw_outcomes = data['OUTCOME'].astype(int).values
        mask_outcome = (raw_outcomes == 1) | (raw_outcomes == 2)
        
        # RE-ENABLE THIS FILTER:
        m1 = data['m1'].values
        m2 = data['m2'].values
        mass_ratio = m1 / (m2 + 1e-9)
        mask_hard = (mass_ratio > 0.2) & (mass_ratio < 5.0)
        
        final_mask = mask_outcome & mask_hard # Combine them
        
        print(f"[Dataset] Filtering: kept {np.sum(final_mask)} 'hard' exchanges.")
        # =================================================================

        df_subset = data[final_mask].copy()
        y_subset = (raw_outcomes[final_mask] == 2).astype(np.float32)

        dfs = [df_subset]
        ys = [y_subset]

        # Augmentation (Redundant for Siamese usually, but option preserved)
        if augment:
            df_aug = df_subset.copy()
            df_aug['m1'], df_aug['m2'] = df_subset['m2'], df_subset['m1']
            df_aug['psi'] += np.pi
            # Label flips 0->1, 1->0
            y_aug = 1.0 - y_subset
            dfs.append(df_aug)
            ys.append(y_aug)
        
        self.df = pd.concat(dfs, ignore_index=True)
        self.y = np.concatenate(ys)
        
        print(f"[Dataset] Final Training Size: {len(self.df)} samples (Aug={augment}).")
        
        # 1. Generate Original State
        self.X_orig = physics_engine.convert_batch_to_state(self.df, align=align)
        
        # 2. Generate Mirror State
        df_mirror = self.df.copy()
        df_mirror['m1'], df_mirror['m2'] = self.df['m2'], self.df['m1']
        df_mirror['psi'] += np.pi 
        self.X_mirror = physics_engine.convert_batch_to_state(df_mirror, align=align)
        
        if scaler:
            self.X_orig = scaler.transform(self.X_orig)
            self.X_mirror = scaler.transform(self.X_mirror)
            self.scaler = scaler
        else:
            self.scaler = StandardScaler()
            combined = np.concatenate([self.X_orig, self.X_mirror], axis=0)
            self.scaler.fit(combined)
            self.X_orig = self.scaler.transform(self.X_orig)
            self.X_mirror = self.scaler.transform(self.X_mirror)
            
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): 
        return (torch.tensor(self.X_orig[idx]), 
                torch.tensor(self.X_mirror[idx]), 
                torch.tensor(self.y[idx]))

# ==========================================
# 3. MODELS & LOSS
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, h, drop):
        super().__init__()
        self.b = nn.Sequential(nn.Linear(h,h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(drop),
                               nn.Linear(h,h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(drop))
    def forward(self, x): return x + self.b(x)

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none', pos_weight=self.pos_weight
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class SymmetricThreeBodyNet(nn.Module):
    """
    Wraps the base model to enforce physical symmetry hard.
    Output = (Model(x) - Model(swap)) / 2
    """
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        
    def forward(self, x, x_mirror):
        out_orig = self.base(x)
        out_mirror = self.base(x_mirror)
        return (out_orig - out_mirror) / 2.0

class ThreeBodyResNet(nn.Module):
    def __init__(self, i_dim, output_dim=1, h=512, n=6, drop=0.2):
        super().__init__()
        self.in_l = nn.Sequential(nn.Linear(i_dim, h), nn.BatchNorm1d(h), nn.GELU())
        self.res = nn.Sequential(*[ResidualBlock(h, drop) for _ in range(n)])
        self.head = nn.Sequential(nn.Linear(h, h//2), nn.GELU(), nn.Linear(h//2, output_dim))
    def forward(self, x): return self.head(self.res(self.in_l(x)))
   
# ==========================================
# 4. OPTIMIZATION (OPTUNA) - UPDATED
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
            print("No database history found and N_TRIALS=0.")
            return {'lr': 1e-3, 'batch_size': 512, 'dropout': 0.2}

    def objective(trial):
        # 1. Suggest Hyperparameters
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        drop = trial.suggest_float("dropout", 0.1, 0.4)
        bs = trial.suggest_categorical("batch_size", [256, 512])
        
        # 2. Fast Data Split (Train/Val)
        subset_size = int(0.25 * len(dataset)) # Use 25% of data for speed
        ds_subset, _ = random_split(dataset, [subset_size, len(dataset)-subset_size])
        
        t_size = int(0.8 * len(ds_subset))
        v_size = len(ds_subset) - t_size
        ds_t, ds_v = random_split(ds_subset, [t_size, v_size])
        
        train_loader = DataLoader(ds_t, batch_size=bs, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
        val_loader = DataLoader(ds_v, batch_size=bs, shuffle=False, num_workers=NUM_WORKERS)
        
        # 3. Setup Model (MATCHING TRAIN_ENSEMBLE)
        # We must wrap it to optimize for the symmetric architecture
        base_model = ThreeBodyResNet(dataset.X_orig.shape[1], 1, 512, 6, drop)
        model = SymmetricThreeBodyNet(base_model).to(device)
        
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        
        # 4. Focal Loss (MATCHING TRAIN_ENSEMBLE)
        pos_weight = torch.tensor([1.0]).to(device) 
        crit = BinaryFocalLoss(alpha=0.25, gamma=2.0, pos_weight=pos_weight)

        # 5. Optimization Loop
        for epoch in range(EPOCHS_OPT):
            model.train()
            for x, x_m, y in train_loader:
                x, x_m, y = x.to(device), x_m.to(device), y.to(device)
                opt.zero_grad()
                
                # Symmetric Forward Pass
                logits = model(x, x_m).squeeze(1)
                
                loss = crit(logits, y.float())
                loss.backward()
                opt.step()
            
            # Pruning (Optional: Stop bad trials early)
            # trial.report(val_loss, epoch)
            # if trial.should_prune(): raise optuna.TrialPruned()

        # 6. Validation
        model.eval()
        val_loss = 0
        count = 0
        with torch.no_grad():
            for x, x_m, y in val_loader:
                x, x_m, y = x.to(device), x_m.to(device), y.to(device)
                
                # Symmetric Forward Pass
                logits = model(x, x_m).squeeze(1)
                
                loss = crit(logits, y.float())
                val_loss += loss.item()
                count += 1
        
        return val_loss / count
        
    study.optimize(objective, n_trials=N_TRIALS)
    return study.best_params

# ==========================================
# 5. TRAINING (ENSEMBLE)
# ==========================================
def train_ensemble(name, dataset, params, device, epochs, n_models=1):
    print(f"\n--- Training {name} ({n_models} Brains) ---")
    
    y_int = dataset.y.astype(int) 
    counts = np.bincount(y_int)
    weights = 1. / (counts + 1e-6)
    global_sampler = WeightedRandomSampler(weights[y_int], len(dataset.y), replacement=True)
    
    lr_history_all = []
    decay_points = [20, 60, 140, 300]
    DECAY_FACTOR = 0.75 
    
    for i in range(n_models):
        print(f"   Brain {i+1}/{n_models}...")

        # --- SPECIALIST MODE (Upgrade 3) ---
        if i >= 5:
            m1 = dataset.df['m1'].values; m2 = dataset.df['m2'].values
            q = m1 / (m2 + 1e-9)
            mask_specialist = (q > 0.5) & (q < 2.0)
            indices = np.where(mask_specialist)[0]
            if len(indices) > 0:
                current_sampler = torch.utils.data.SubsetRandomSampler(indices)
                print(f"      [Specialist Mode] Training on {len(indices)} HARD samples.")
            else:
                current_sampler = global_sampler
        else:
            current_sampler = global_sampler
            
        loader = DataLoader(dataset, batch_size=params['batch_size'], sampler=current_sampler, num_workers=NUM_WORKERS, drop_last=True)
        
        # --- SYMMETRIC MODEL (Upgrade 2) ---
        base_model = ThreeBodyResNet(dataset.X_orig.shape[1], 1, 512, 6, params['dropout'])
        model = SymmetricThreeBodyNet(base_model).to(device)
        
        opt = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=1e-4)
        sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)
        
        # --- FOCAL LOSS (Upgrade 1) ---
        pos_weight = torch.tensor([1.0]).to(device) 
        crit = BinaryFocalLoss(alpha=0.25, gamma=2.0, pos_weight=pos_weight)
        
        ckpt_path = f"{MODEL_S3_PREFIX}_{i}_checkpoint.pth"
        start_epoch = 0
        lr_track = []
        
        if os.path.exists(ckpt_path):
            print(f"      >> Resuming from checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint['model_state'])
            opt.load_state_dict(checkpoint['optimizer_state'])
            sched.load_state_dict(checkpoint['scheduler_state'])
            start_epoch = checkpoint['epoch'] + 1
            lr_track = checkpoint.get('lr_history', [])
            if 'base_lrs' in checkpoint: sched.base_lrs = checkpoint['base_lrs']

        for epoch in range(start_epoch, epochs):
            model.train()
            current_lr = opt.param_groups[0]['lr']
            lr_track.append(current_lr)
            total_loss = 0
            
            for x, x_m, y in loader:
                x, x_m, y = x.to(device), x_m.to(device), y.to(device)
                opt.zero_grad()
                logits = model(x, x_m).squeeze(1)
                loss = crit(logits, y.float())
                loss.backward()
                opt.step()
                total_loss += loss.item()
            
            sched.step()

            if (epoch + 1) in decay_points:
                print(f"     [Auto-Decay] Reducing restart spike by {DECAY_FACTOR}x")
                sched.base_lrs = [lr * DECAY_FACTOR for lr in sched.base_lrs]
                for param_group, new_lr in zip(opt.param_groups, sched.base_lrs):
                    param_group['lr'] = new_lr
            
            if (epoch+1) % 10 == 0:
                avg_loss = total_loss/len(loader)
                print(f"      Ep {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | LR: {current_lr:.1e}")
                torch.save({
                    'epoch': epoch, 'model_state': model.state_dict(), 
                    'optimizer_state': opt.state_dict(), 'scheduler_state': sched.state_dict(),
                    'base_lrs': sched.base_lrs, 'lr_history': lr_track
                }, ckpt_path)
                
        torch.save(model.state_dict(), f"{MODEL_S3_PREFIX}_{i}.pth")
        if os.path.exists(ckpt_path): os.remove(ckpt_path) 
        if i == 0: lr_history_all = lr_track
    return lr_history_all

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    physics = ThreeBodyPhysics()
    
    # ==========================================
    # 1. OPTIMIZATION
    # ==========================================
    print("\n--- Hyperparameter Setup ---")
    # CRITICAL FIX: Using 'CascadeSiameseDataset' (your existing class name)
    # Ensure this class has the 'Hard Mode' filter enabled in its __init__
    ds_opt = CascadeSiameseDataset(TRAIN_FILE, physics, align=ALIGN_TO_BINARY, augment=False)
    
    p_s3 = run_optimization("opt_s3_siamese_v2", ds_opt, device)
    print("Params:", p_s3)
    
    # ==========================================
    # 2. TRAINING (Ensemble of Hard-Mode Brains)
    # ==========================================
    scaler = ds_opt.scaler
    # This dataset MUST produce only "Hard" cases for the NN to learn correctly
    ds_train = CascadeSiameseDataset(TRAIN_FILE, physics, scaler=scaler, align=ALIGN_TO_BINARY, augment=AUGMENT_TRAIN)
    
    lr_hist = train_ensemble("Stage 3 Siamese", ds_train, p_s3, device, EPOCHS_S3, n_models=N_MODELS_S3)

    # Plot & Save LR History
    plt.figure(figsize=(10, 4))
    plt.plot(lr_hist)
    plt.title("Learning Rate (Cosine Annealing)")
    plt.xlabel("Epochs")
    plt.ylabel("LR")
    plt.savefig(f"{MODEL_S3_PREFIX}_lr_history.png")
    plt.show()

    # ==========================================
    # 3. EVALUATION (Hybrid: Rule + AI)
    # ==========================================
    print("\n--- Evaluation (Hybrid: Rule + AI) ---")
    df_test = pd.read_csv(TEST_FILE, sep=r'\s+', engine='python')
    
    # Filter for exchanges
    mask_ex = (df_test['OUTCOME'] == 1) | (df_test['OUTCOME'] == 2)
    df_test_ex = df_test[mask_ex].copy()
    
    y_true = (df_test_ex['OUTCOME'] == 2).astype(int).values 
    
    # A. PREPARE DATA
    m1_test = df_test_ex['m1'].values
    m2_test = df_test_ex['m2'].values
    mass_ratios = m1_test / (m2_test + 1e-9)
    
    # Identify Hard vs Easy cases
    mask_test_hard = (mass_ratios > 0.2) & (mass_ratios < 5.0)
    print(f"Test Set Split: {np.sum(mask_test_hard)} Hard Cases, {np.sum(~mask_test_hard)} Easy Cases.")

    # Convert to tensors
    X_test_all = torch.tensor(scaler.transform(physics.convert_batch_to_state(df_test_ex, align=ALIGN_TO_BINARY)), dtype=torch.float32).to(device)
    
    # Mirror state for Symmetry
    df_swap = df_test_ex.copy()
    df_swap['m1'], df_swap['m2'] = df_swap['m2'], df_swap['m1']; df_swap['psi'] += np.pi
    X_test_mirror_all = torch.tensor(scaler.transform(physics.convert_batch_to_state(df_swap, align=ALIGN_TO_BINARY)), dtype=torch.float32).to(device)
    
    y_probs_final = np.zeros(len(df_test_ex))
    
    # --- LOGIC PART 1: EASY CASES (PHYSICS RULE) ---
    y_probs_final[~mask_test_hard] = 0.0
    mask_easy_class1 = (mass_ratios <= 0.2)
    y_probs_final[mask_easy_class1] = 1.0
    
    # --- LOGIC PART 2: THE HARD CASES (NEURAL NETWORK) ---
    if np.sum(mask_test_hard) > 0:
        print("Evaluating Hard Cases with Neural Network...")
        X_hard = X_test_all[mask_test_hard]
        X_hard_mirror = X_test_mirror_all[mask_test_hard]
        
        vote_sum_hard = np.zeros(len(X_hard))
        active_models = 0
        
        for i in range(N_MODELS_S3):
            fname = f"{MODEL_S3_PREFIX}_{i}.pth"
            if not os.path.exists(fname): continue
            
            # --- FIX: Re-create the structure exactly as trained ---
            base_model = ThreeBodyResNet(X_hard.shape[1], 1, 512, 6, p_s3['dropout'])
            model = SymmetricThreeBodyNet(base_model) # Wrap it!
            
            # Now load state dict (it matches the wrapper structure)
            model.load_state_dict(torch.load(fname))
            model.to(device)
            model.eval()
            
            with torch.no_grad():
                # --- FIX: Pass BOTH views to the model ---
                logits = model(X_hard, X_hard_mirror).squeeze(1)
                p_comb = torch.sigmoid(logits)
                
                vote_sum_hard += p_comb.cpu().numpy()
                active_models += 1
        
        if active_models > 0:
            y_probs_final[mask_test_hard] = vote_sum_hard / active_models
        else:
            print("WARNING: No active models found for hard cases!")

    y_probs = y_probs_final
    
    # --- LOGIC PART 1: THE EASY CASES (PHYSICS RULE) ---
    # Rule: If mass ratio is extreme (>3 or <0.33), the heavy star keeps the intruder.
    # In this dataset format, ejecting the lighter star (m2) is Outcome 1 (Class 0).
    # So we force probability to 0.0.
    y_probs_final[~mask_test_hard] = 0.0
    mask_easy_class1 = (mass_ratios <= 0.2)
    y_probs_final[mask_easy_class1] = 1.0
    
    # --- LOGIC PART 2: THE HARD CASES (NEURAL NETWORK) ---
    if np.sum(mask_test_hard) > 0:
        print("Evaluating Hard Cases with Neural Network...")
        X_hard = X_test_all[mask_test_hard]
        X_hard_mirror = X_test_mirror_all[mask_test_hard]
        
        vote_sum_hard = np.zeros(len(X_hard))
        active_models = 0
        
        for i in range(N_MODELS_S3):
            fname = f"{MODEL_S3_PREFIX}_{i}.pth"
            if not os.path.exists(fname): continue
                
            model = ThreeBodyResNet(X_hard.shape[1], 1, 512, 6, p_s3['dropout'])
            model.load_state_dict(torch.load(fname))
            model.to(device)
            model.eval()
            
            with torch.no_grad():
                # TTA Logic: Average Original and Mirror predictions
                l_orig = model(X_hard).squeeze(1)
                p_orig = torch.sigmoid(l_orig)
                
                l_mirr = model(X_hard_mirror).squeeze(1)
                p_mirr = torch.sigmoid(l_mirr)
                
                p_comb = (p_orig + (1.0 - p_mirr)) / 2.0
                vote_sum_hard += p_comb.cpu().numpy()
                active_models += 1
        
        if active_models > 0:
            y_probs_final[mask_test_hard] = vote_sum_hard / active_models
        else:
            print("WARNING: No active models found for hard cases!")

    y_probs = y_probs_final
    
    # ==========================================
    # 4. METRICS & PLOTTING
    # ==========================================
    THRESHOLD = 0.42
    y_pred = (y_probs > THRESHOLD).astype(int)
    
    # Histogram
    plt.figure(figsize=(7,4))
    plt.hist(y_probs, bins=50, alpha=0.7, color='purple')
    plt.title("Distribution of Predicted Probabilities (Hybrid)")
    plt.xlabel("Probability (0 = Exch 1-3, 1 = Exch 2-3)")
    plt.ylabel("Count")
    plt.axvline(THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Threshold ({THRESHOLD})')
    plt.legend()
    plt.savefig(f"{MODEL_S3_PREFIX}_histogram.png")
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    acc = (cm[0,0] + cm[1,1]) / np.sum(cm)
    print(f"\nEnsemble Accuracy: {acc*100:.2f}%")
    
    # Counts Plot
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Exch 1-3', 'Exch 2-3'], 
                yticklabels=['Exch 1-3', 'Exch 2-3'])
    plt.title(f"Confusion Matrix (Counts) - Acc: {acc*100:.1f}%")
    plt.ylabel('True Outcome')
    plt.xlabel('Predicted Outcome')
    plt.savefig(f"{MODEL_S3_PREFIX}_cm_counts.png")
    plt.show()

    # Normalized Plot
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=['Exch 1-3', 'Exch 2-3'], 
                yticklabels=['Exch 1-3', 'Exch 2-3'])
    plt.title("Confusion Matrix (Normalized)")
    plt.ylabel('True Outcome')
    plt.xlabel('Predicted Outcome')
    plt.savefig(f"{MODEL_S3_PREFIX}_cm_normalized.png")
    plt.show()
    
    # ==========================================
    # 5. ADVANCED DIAGNOSTICS (Physics Breakdown)
    # ==========================================
    print("\n--- Running Advanced Physics Diagnostics ---")

    
    # 1. PHYSICS ACCURACY PLOTS
    # We want to see how accuracy changes vs. Mass Ratio and Eccentricity
    
    # Helper function to plot binned accuracy
    def plot_binned_accuracy(variable, var_name, y_true, y_pred, bins=10):
        # Create bins
        bin_edges = np.linspace(variable.min(), variable.max(), bins+1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        accuracies = []
        counts = []
        
        for i in range(bins):
            # Mask for current bin
            mask = (variable >= bin_edges[i]) & (variable < bin_edges[i+1])
            if np.sum(mask) > 0:
                # Calculate accuracy in this bin
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
        
        # Add a histogram in the background to show where most data is
        ax2 = plt.gca().twinx()
        ax2.bar(bin_centers, counts, width=(bin_edges[1]-bin_edges[0])*0.9, alpha=0.1, color='gray')
        ax2.set_ylabel("Count of Samples")
        
        plt.savefig(f"{MODEL_S3_PREFIX}_acc_vs_{var_name}.png")
        plt.show()

    # Define variables from the test dataframe
    mass_ratio_val = df_test_ex['m1'].values / (df_test_ex['m2'].values + 1e-9)
    eccentricity_val = df_test_ex['e'].values
    impact_param_val = df_test_ex['b_pc'].values
    
    # Generate the Plots
    # Only look at the "Hard" cases for these plots, as Easy cases are 100% correct
    if np.sum(mask_test_hard) > 0:
        y_true_hard = y_true[mask_test_hard]
        y_pred_hard = y_pred[mask_test_hard]
        
        # Plot 1: Accuracy vs Mass Ratio (The most critical one)
        plot_binned_accuracy(mass_ratio_val[mask_test_hard], "Mass Ratio (q)", y_true_hard, y_pred_hard)
        
        # Plot 2: Accuracy vs Eccentricity
        plot_binned_accuracy(eccentricity_val[mask_test_hard], "Eccentricity (e)", y_true_hard, y_pred_hard)

    # 2. FAILURE MAP (Where are the errors?)
    # Scatter plot: Mass Ratio vs Impact Parameter
    # Red X = Wrong, Green O = Correct
    if np.sum(mask_test_hard) > 0:
        plt.figure(figsize=(8, 6))
        correct_mask = (y_true == y_pred)
        
        # Plot Correct (Small dots)
        plt.scatter(mass_ratio_val[correct_mask & mask_test_hard], 
                    impact_param_val[correct_mask & mask_test_hard], 
                    c='green', s=10, alpha=0.3, label='Correct')
        
        # Plot Incorrect (Larger X)
        plt.scatter(mass_ratio_val[~correct_mask & mask_test_hard], 
                    impact_param_val[~correct_mask & mask_test_hard], 
                    c='red', marker='x', s=30, alpha=0.6, label='Wrong')
        
        plt.title("Failure Map: Mass Ratio vs Impact Parameter")
        plt.xlabel("Mass Ratio (m1/m2)")
        plt.ylabel("Impact Parameter (b)")
        plt.legend()
        plt.savefig(f"{MODEL_S3_PREFIX}_failure_map.png")
        plt.show()

    # 3. ROC CURVE (Evaluating the Blob)
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(f"{MODEL_S3_PREFIX}_roc.png")
    plt.show()
    
    print(f"Diagnostics Complete. ROC AUC: {roc_auc:.3f}")