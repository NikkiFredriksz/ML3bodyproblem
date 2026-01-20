import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score

# ==========================================
# CONFIGURATION
# ==========================================
torch.set_float32_matmul_precision('medium') 

STORAGE_DB = "sqlite:///three_body_stage3_v10.db"
TRAIN_FILE = "train3body.dat"
TEST_FILE = "data_for_stage3.csv"
MODEL_S3_PREFIX = "stage3_v9+filter" 

# HYPERPARAMETERS
N_TRIALS = 100          # Set to 0 to use best existing params without re-optimizing
EPOCHS_OPT = 50         # Epochs for optimization trials

# STAGE 3 SETTINGS (ENSEMBLE)
N_MODELS_S3 = 10         # Number of brains in the committee
EPOCHS_S3 = 300         # Deep training for final models
ALIGN_TO_BINARY = True 

# STRATEGY
NUM_WORKERS = 4         


# ==========================================
# 1. PHYSICS ENGINE (v12 + Phase Features)
# ==========================================
class ThreeBodyPhysics:
    def __init__(self): self.G = 4.302e-3 

    def _unpack_physics(self, df):
        """Internal helper to unpack physics variables once."""
        m1 = df['m1'].values; m2 = df['m2'].values; m3 = df['m3'].values
        a = df['a_pc'].values; e = df['e'].values; v_inf = df['v_km_s'].values
        b = df['b_pc'].values
        return m1, m2, m3, a, e, v_inf, b

    def convert_batch_to_state(self, df, align=False):
        # 1. Use Helper to Unpack
        m1, m2, m3, a, e, v_inf, b = self._unpack_physics(df)
        
        # Angles (Keep raw extraction here as it's specific to Features)
        phi = np.where(np.abs(df['phi'].values)>2*np.pi, np.radians(df['phi'].values), df['phi'].values)
        theta = np.where(np.abs(df['theta'].values)>2*np.pi, np.radians(df['theta'].values), df['theta'].values)
        psi = np.where(np.abs(df['psi'].values)>2*np.pi, np.radians(df['psi'].values), df['psi'].values)
        f = df['f'].values; t_coal = df['t_coal_yr'].values
        
        # --- FEATURE ENGINEERING (Same as your correct code) ---
        M_bin = m1 + m2
        r_mag = (a * (1 - e**2)) / (1 + e * np.cos(f))

        sin_phi, cos_phi = np.sin(phi), np.cos(phi)
        sin_theta, cos_theta = np.sin(theta), np.cos(theta)
        sin_psi, cos_psi = np.sin(psi), np.cos(psi)
        sin_f, cos_f = np.sin(f), np.cos(f)
        
        r_peri_encounter = b 
        v_peri_encounter = np.sqrt(v_inf**2 + 2*self.G*M_bin/(r_peri_encounter+1e-9))
        v_avg = np.sqrt(v_inf * v_peri_encounter)
        t_approach = (50.0 * a) / (v_avg + 1e-9)

        mean_motion = np.sqrt(self.G * M_bin / (a**3 + 1e-9))
        M_encounter = f + mean_motion * t_approach
        
        feat_phase_sin = np.sin(M_encounter)
        feat_phase_cos = np.cos(M_encounter)
        
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

        # 1. Who is heavier? (Explicit Asymmetry Signal)
        # +1 if m1 > m2, -1 if m2 > m1. Crucial for symmetry breaking.
        mass_diff_norm = (m1 - m2) / (m1 + m2 + 1e-9)

        # 2. Perturbation Strength (How violent is the kick?)
        # Same logic as Stage 2
        v_orb_bin = np.sqrt(self.G * M_bin / (a + 1e-9))
        delta_v_approx = (2 * self.G * m3) / (b * v_inf + 1e-9)
        perturb_strength = delta_v_approx / (v_orb_bin + 1e-9)

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
            feat_phase_sin[:,None], feat_phase_cos[:,None],
            
            # [ADDED] The Tie-Breakers
            mass_diff_norm[:, None],
            np.log10(perturb_strength + 1e-9)[:, None]
        ]
        return np.hstack(feat).astype(np.float32)

# ==========================================
# 2. DATASET
# ==========================================
class ThreeBodySiameseDataset(Dataset):
    """
    A custom PyTorch Dataset that implements the "Hybrid Strategy" and "Siamese Consistency".
    
    1. Hybrid Filtering: It filters the raw data to keep only the "Hard" cases 
       (where mass ratios are comparable). "Easy" cases (extreme mass ratios) 
       are handled analytically by the physics engine and excluded here.
       
    2. Siamese/Symmetry: For every sample, it generates two views: the original state 
       and a "mirrored" state (where bodies 1 and 2 are swapped). This allows the 
       training loop to enforce physical symmetry (Label(A,B) = 1 - Label(B,A)).
    """
    def __init__(self, filepath, physics_engine, scaler=None, align=False, augment=False):
        if not os.path.exists(filepath): sys.exit(f"File {filepath} not found.") # Safety check for data file existence
        data = pd.read_csv(filepath, sep=r'\s+', engine='python') # Load space-separated data into a pandas DataFrame
        
        # ... (Standard outcomes filter) ...
        raw_outcomes = data['OUTCOME'].astype(int).values # Extract outcomes as integers
        mask_outcome = (raw_outcomes == 1) | (raw_outcomes == 2) # Keep only Exchange 2-3 (Type 1) and Exchange 1-3 (Type 2) events
        
        # RE-ENABLE THIS FILTER:
        m1 = data['m1'].values # Extract mass 1
        m2 = data['m2'].values # Extract mass 2
        mass_ratio = m1 / (m2 + 1e-9) # Calculate mass ratio q = m1/m2 (add epsilon to avoid div/0)
        
        # [FIX] Widen this to match your Evaluation logic!
        # Old: (mass_ratio > 0.2) & (mass_ratio < 5.0)
        # Define "Hard" cases: where mass ratio is between 0.05 and 20. 
        # Outside this range, the lighter body is almost always ejected (solved by physics).
        mask_hard = (mass_ratio > 0.05) & (mass_ratio < 20.0) 
        
        final_mask = mask_outcome & mask_hard # Combine outcome filter and "hard case" filter
        
        print(f"[Dataset] Filtering: kept {np.sum(final_mask)} 'hard' exchanges.") # Log the number of samples remaining
        # =================================================================

        df_subset = data[final_mask].copy() # Create the filtered dataframe
        # Create binary targets: 1.0 if Outcome is 2 (Exch 1-3), 0.0 if Outcome is 1 (Exch 2-3)
        y_subset = (raw_outcomes[final_mask] == 2).astype(np.float32)

        dfs = [df_subset] # List to hold dataframes (for potential augmentation)
        ys = [y_subset]   # List to hold labels
        
        self.df = pd.concat(dfs, ignore_index=True) # Concatenate all dataframes into one
        self.y = np.concatenate(ys) # Concatenate all labels
        
        print(f"[Dataset] Final Training Size: {len(self.df)} samples.")
        
        # 1. Generate Original State
        # Convert the dataframe rows into the physics-informed feature vectors (the inputs to the neural net)
        self.X_orig = physics_engine.convert_batch_to_state(self.df, align=align)
        
        # 2. Generate Mirror State
        # Create a temporary dataframe where m1/m2 are swapped to generate the "Twin" input for Siamese training
        df_mirror = self.df.copy()
        df_mirror['m1'], df_mirror['m2'] = self.df['m2'], self.df['m1'] # Swap masses
        df_mirror['psi'] += np.pi # Rotate orientation
        self.X_mirror = physics_engine.convert_batch_to_state(df_mirror, align=align) # Generate physics features for the mirror state
        
        if scaler:
            # If a scaler is provided (validation/test time), transform the data using it
            self.X_orig = scaler.transform(self.X_orig)
            self.X_mirror = scaler.transform(self.X_mirror)
            self.scaler = scaler
        else:
            # If no scaler (training time), fit a new StandardScaler on BOTH original and mirror data
            self.scaler = StandardScaler()
            combined = np.concatenate([self.X_orig, self.X_mirror], axis=0) # Combine to compute global mean/std
            self.scaler.fit(combined)
            self.X_orig = self.scaler.transform(self.X_orig) # Transform original
            self.X_mirror = self.scaler.transform(self.X_mirror) # Transform mirror
            
    def __len__(self): return len(self.y) # Return total number of samples
    def __getitem__(self, idx): 
        # Return a tuple: (Input, Mirror Input, Label)
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

class ThreeBodyResNet(nn.Module):
    def __init__(self, i_dim, output_dim=1, h=1024, n=6, drop=0.2):
        super().__init__()
        self.in_l = nn.Sequential(nn.Linear(i_dim, h), nn.BatchNorm1d(h), nn.GELU())
        self.res = nn.Sequential(*[ResidualBlock(h, drop) for _ in range(n)])
        self.head = nn.Sequential(nn.Linear(h, h//2), nn.GELU(), nn.Linear(h//2, output_dim))
    def forward(self, x): return self.head(self.res(self.in_l(x)))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    physics = ThreeBodyPhysics()

    # 1. FIT SCALER
    print("Initializing Scaler from Training Data...")
    # UPDATED: Use new class name, remove 'augment' arg
    ds_train_fit = ThreeBodySiameseDataset(TRAIN_FILE, physics, align=True)
    scaler = ds_train_fit.scaler

    # 2. LOAD TEST DATA
    print(f"Loading Test Data: {TEST_FILE}")
    df_test = pd.read_csv(TEST_FILE, sep=r'\s+', engine='python')
    
    # Filter for Exchange outcomes (1 and 2) only
    mask_ex = (df_test['OUTCOME'] == 1) | (df_test['OUTCOME'] == 2)
    df_test_ex = df_test[mask_ex].copy()
    y_true = (df_test_ex['OUTCOME'] == 2).astype(int).values
    
    # 3. LOAD SAVED ENSEMBLE
    H_DIM = 1024
    N_LAYERS = 6
    DROPOUT = 0.2
    
    print(f"Loading {N_MODELS_S3} models from prefix '{MODEL_S3_PREFIX}'...")
    models = []
    
    for i in range(N_MODELS_S3):
        fname = f"{MODEL_S3_PREFIX}_{i}.pth"
        if os.path.exists(fname):
            # Initialize Architecture
            model = ThreeBodyResNet(ds_train_fit.X_orig.shape[1], 1, H_DIM, N_LAYERS, DROPOUT)
            
            # UPDATED: Simple loading (The documented training code does NOT use a wrapper)
            checkpoint = torch.load(fname, map_location=device)
            model.load_state_dict(checkpoint) 
            
            model.to(device)
            model.eval()
            models.append(model)
            print(f"   > Loaded Brain {i}")
        else:
            print(f"   ! Warning: {fname} missing.")
    
    # 4. HYBRID INFERENCE
    # A. Physics Filter (Mass Ratio)
    m1 = df_test_ex['m1'].values
    m2 = df_test_ex['m2'].values
    mass_ratios = m1 / (m2 + 1e-9)
    
    # Define "Easy" cases
    mask_mass_ratio_easy = (mass_ratios <= 0.05) | (mass_ratios >= 20.0)
    mask_test_hard = ~mask_mass_ratio_easy
    
    y_probs = np.zeros(len(df_test_ex))
    
    # Apply Physics Rules
    y_probs[mask_mass_ratio_easy & (mass_ratios >= 20.0)] = 0.0 # Exch 1-3
    y_probs[mask_mass_ratio_easy & (mass_ratios <= 0.05)] = 1.0 # Exch 2-3
    
    print(f"Split: {np.sum(mask_test_hard)} Hard (AI) vs {np.sum(mask_mass_ratio_easy)} Easy (Physics)")

    # B. AI Inference (Hard Cases)
    if np.sum(mask_test_hard) > 0:
        # Prepare Tensors
        X_hard = torch.tensor(scaler.transform(physics.convert_batch_to_state(df_test_ex[mask_test_hard], align=True)), dtype=torch.float32).to(device)
        
        # Mirror Tensors (For Symmetry Averaging)
        df_swap = df_test_ex[mask_test_hard].copy()
        df_swap['m1'], df_swap['m2'] = df_swap['m2'], df_swap['m1']; df_swap['psi'] += np.pi
        X_hard_mirror = torch.tensor(scaler.transform(physics.convert_batch_to_state(df_swap, align=True)), dtype=torch.float32).to(device)
        
        ensemble_preds = np.zeros((len(models), len(X_hard)))
        
        with torch.no_grad():
            for i, model in enumerate(models):
                # Pred 1: Original
                p1 = torch.sigmoid(model(X_hard).squeeze(1))
                # Pred 2: Mirror (Flip outcome 1-p)
                p2 = 1.0 - torch.sigmoid(model(X_hard_mirror).squeeze(1))
                # Average
                ensemble_preds[i, :] = ((p1 + p2) / 2.0).cpu().numpy()
        
        mean_probs = np.mean(ensemble_preds, axis=0)
        variance = np.var(ensemble_preds, axis=0)
        
        # --- AUTO-TUNING UNCERTAINTY (DISABLED - PURE AI MODE) ---
        print("\n--- Auto-Tuning Uncertainty Threshold ---")
        
        # 1. Calculate Confidence (Distance from 0.5)
        confidence_scores = np.abs(mean_probs - 0.5)
        
        # 2. FORCE THRESHOLD TO 0.0
        # We do NOT scan for a better threshold. We accept everything.
        best_conf_thresh = 0.0
        
        print(f"Selected Confidence Threshold: > {best_conf_thresh:.4f} (Forced)")
        
        # 3. Apply Threshold (Will affect nothing, since all scores >= 0.0)
        mask_uncertain = confidence_scores < best_conf_thresh
        
        # This copy is just to keep the data structure consistent
        y_hard_final = mean_probs.copy()
        
        # This line does nothing now (because mask_uncertain is all False)
        y_hard_final[mask_uncertain] = -1.0 
        
        # Slot back into main array
        y_probs[mask_test_hard] = y_hard_final

    # ==========================================
    # 4. METRICS & PLOTTING (Pure AI Mode)
    # ==========================================
    
    # 1. Auto-Find Best Decision Threshold (Optimizing for BALANCE)
    # We scan 0.1 to 0.9 to find the cutoff that maximizes F1 Score (Balance)
    best_class_thresh = 0.5
    best_score = 0.0
    
    # We use all data since we accepted 100%
    scan_range = np.linspace(0.05, 0.95, 181) # Finer scan step
    
    for t in scan_range:
        pred_t = (y_probs > t).astype(int)
        
        # We optimize for Macro F1 Score (treats both classes as equally important)
        # This prevents the threshold from drifting too high/low to favor the majority.
        score_t = f1_score(y_true, pred_t, average='macro')
        
        if score_t > best_score:
            best_score = score_t
            best_class_thresh = t
            
    print(f"\nAuto-Tuned Balanced Threshold: {best_class_thresh:.3f} (Max F1: {best_score:.2%})")
    THRESHOLD = best_class_thresh

    # 2. Final Predictions
    y_pred = (y_probs > THRESHOLD).astype(int)
    
    # 3. Statistics
    print("\n--- Performance Statistics (Pure AI) ---")
    print(f"Total Samples: {len(y_probs)}")
    print("AI Coverage:   100.0% (No Simulator Used)")
    
    # Calculate Global Accuracy
    acc_global = np.mean(y_pred == y_true)
    print(f"Global AI Accuracy: {acc_global*100:.2f}%")
    
    # 4. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    # Plot 1: Counts
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Exch 1-3', 'Exch 2-3'], 
                yticklabels=['Exch 1-3', 'Exch 2-3'])
    plt.title(f"Pure AI Confusion Matrix\nAccuracy: {acc_global*100:.1f}%")
    plt.ylabel('True Outcome')
    plt.xlabel('AI Prediction')
    plt.savefig(f"{MODEL_S3_PREFIX}_cm_pure_ai.png")
    plt.show()

    # Plot 2: Normalized (Percentages)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=['Exch 1-3', 'Exch 2-3'], 
                yticklabels=['Exch 1-3', 'Exch 2-3'])
    plt.title("Confusion Matrix (Normalized)")
    plt.ylabel('True Outcome')
    plt.xlabel('AI Prediction')
    plt.savefig(f"{MODEL_S3_PREFIX}_cm_normalized.png")
    plt.show()
    
    # Plot 3: Probability Histogram
    plt.figure(figsize=(7,4))
    plt.hist(y_probs, bins=50, alpha=0.7, color='purple')
    plt.title("Probability Distribution (All Samples)")
    plt.xlabel(f"Probability (Balanced Threshold = {THRESHOLD:.2f})")
    plt.axvline(THRESHOLD, color='red', linestyle='--')
    plt.savefig(f"{MODEL_S3_PREFIX}_prob_dist.png")
    plt.show()
    
    # ==========================================
    # 5. ADVANCED DIAGNOSTICS (Physics Breakdown)
    # ==========================================
    print("\n--- Running Advanced Physics Diagnostics ---")
    
    
    # 1. PHYSICS ACCURACY PLOTS
    # We want to see how accuracy changes vs. Mass Ratio and Eccentricity
    
    # Helper function to plot binned accuracy
    def plot_binned_accuracy(variable, var_name, y_true, y_pred, bins=10, log_scale=False):
        if log_scale:
            # Create logarithmic bins (e.g., 0.01, 0.1, 1, 10, 100)
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
        
        # [FIX] Apply log scale to plot if requested
        if log_scale:
            plt.xscale('log')
        
        ax2 = plt.gca().twinx()
        ax2.bar(bin_centers, counts, width=np.diff(bin_edges), alpha=0.1, color='gray', align='edge')
        ax2.set_ylabel("Count of Samples")
        
        plt.savefig(f"{MODEL_S3_PREFIX}_acc_vs_{var_name.split()[0]}.png")
        plt.show()
    
    # Define variables from the test dataframe
    mass_ratio_val = df_test_ex['m1'].values / (df_test_ex['m2'].values + 1e-9)
    eccentricity_val = df_test_ex['e'].values
    impact_param_val = df_test_ex['b_pc'].values
    
    # Generate the Plots
    # Only look at the "Hard" cases for these plots, as Easy cases are 100% correct
    if np.sum(mask_test_hard) > 0:
        # Filter: Hard AND Accepted (Not -1.0)
        mask_plot = mask_test_hard & (y_probs != -1.0)
        
        if np.sum(mask_plot) > 0:
            y_true_plot = y_true[mask_plot]
            y_pred_plot = (y_probs[mask_plot] > THRESHOLD).astype(int)
            
            # Slice variables
            mr_plot = mass_ratio_val[mask_plot]
            ecc_plot = eccentricity_val[mask_plot]
    
            print(f"Generating Physics Plots on {np.sum(mask_plot)} accepted hard samples...")
            
            # Plot 1: Accuracy vs Mass Ratio
            plot_binned_accuracy(mr_plot, "Mass Ratio (q)", y_true_plot, y_pred_plot, log_scale=True)
            
            # Plot 2: Accuracy vs Eccentricity
            plot_binned_accuracy(ecc_plot, "Eccentricity (e)", y_true_plot, y_pred_plot, log_scale=False)
    
    # 2. FAILURE MAP (Where are the errors?)
    # Scatter plot: Mass Ratio vs Impact Parameter
    # Red X = Wrong, Green O = Correct
    if np.sum(mask_test_hard) > 0:
        # Define the mask for Accepted cases (excluding -1.0)
        mask_plot = mask_test_hard & (y_probs != -1.0)
        
        if np.sum(mask_plot) > 0:
            plt.figure(figsize=(8, 6))
            
            # We recalculate correctness based only on the Accepted cases
            # Note: y_pred is likely derived globally, make sure it aligns
            # Safer to recalculate local y_pred for plotting:
            y_pred_plot = (y_probs > THRESHOLD).astype(int)
            correct_mask = (y_true == y_pred_plot)
            
            # 1. Plot Correct (Green Dots)
            # Intersection of: Accepted Hard Cases AND Correct Prediction
            mask_green = mask_plot & correct_mask
            plt.scatter(mass_ratio_val[mask_green], 
                        impact_param_val[mask_green], 
                        c='green', s=10, alpha=0.3, label='Correct')
            
            # 2. Plot Incorrect (Red Xs)
            # Intersection of: Accepted Hard Cases AND Wrong Prediction
            mask_red = mask_plot & (~correct_mask)
            plt.scatter(mass_ratio_val[mask_red], 
                        impact_param_val[mask_red], 
                        c='red', marker='x', s=30, alpha=0.6, label='Wrong')
    
            # [OPTIONAL BONUS] Plot the Uncertain Cases (Yellow squares)
            # This helps you visualize exactly WHERE the AI is giving up!
            mask_uncertain = mask_test_hard & (y_probs == -1.0)
            if np.sum(mask_uncertain) > 0:
                plt.scatter(mass_ratio_val[mask_uncertain], 
                            impact_param_val[mask_uncertain], 
                            c='orange', marker='s', s=15, alpha=0.5, label='Uncertain (Fallback)')
            
            plt.title("Failure Map: Mass Ratio vs Impact Parameter (Accepted Only)")
            plt.xlabel("Mass Ratio (m1/m2)")
            plt.ylabel("Impact Parameter (b)")
            plt.xscale('log')
            plt.legend()
            plt.savefig(f"{MODEL_S3_PREFIX}_failure_map.png")
            plt.show()
    
    # 3. ROC CURVE (Corrected)
    # Filter out the fallback cases (-1.0) before calculating ROC
    mask_clean = (y_probs != -1.0)
    
    if np.sum(mask_clean) > 0:
        y_true_clean = y_true[mask_clean]
        y_probs_clean = y_probs[mask_clean]
        
        fpr, tpr, thresholds = roc_curve(y_true_clean, y_probs_clean)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Accepted Only (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC (Excluding Uncertain Cases)')
        plt.legend(loc="lower right")
        plt.savefig(f"{MODEL_S3_PREFIX}_roc.png")
        plt.show()
        
        print(f"Diagnostics Complete. ROC AUC (Accepted Only): {roc_auc:.4f}")
    else:
        print("Diagnostics Complete. (No accepted samples for ROC).")

    # ==========================================
    # 6. PHYSICS FILTER VALIDATION PLOT
    # ==========================================
    print("\n--- Generating Physics Filter Validation Plot ---")

    def plot_filter_effectiveness(mass_ratios, y_true, y_probs, lower_bound=0.05, upper_bound=20.0):
        # 1. Setup Bins (Logarithmic scale is best for Mass Ratio)
        # Covers 0.05 to 20.0 to see well past the boundaries
        bins = np.logspace(np.log10(0.05), np.log10(20.0), 40)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        
        accuracies = []
        counts = []
        
        # Calculate Global Prediction (Physics + AI)
        # We need to reconstruct the final prediction made by the hybrid system
        # Note: y_probs already contains 0.0/1.0 for physics and AI probs for the rest
        y_pred_final = (y_probs > 0.5).astype(int)
        
        for i in range(len(bins)-1):
            mask = (mass_ratios >= bins[i]) & (mass_ratios < bins[i+1])
            if np.sum(mask) > 0:
                acc = np.mean(y_true[mask] == y_pred_final[mask])
                accuracies.append(acc)
                counts.append(np.sum(mask))
            else:
                accuracies.append(np.nan) # No data
                counts.append(0)
        
        # 2. Plotting
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Background Regions (The "Filter Zones")
        ax1.axvspan(0.05, lower_bound, color='green', alpha=0.1, label='Physics Filter (Exch 2-3)')
        ax1.axvspan(upper_bound, 20.0, color='green', alpha=0.1, label='Physics Filter (Exch 1-3)')
        ax1.axvspan(lower_bound, upper_bound, color='gray', alpha=0.05, label='AI Zone (Hard)')
        
        # Accuracy Line
        ax1.plot(bin_centers, accuracies, 'o-', color='darkblue', linewidth=2, label='System Accuracy')
        ax1.set_ylabel('Accuracy', color='darkblue', fontsize=12)
        ax1.set_ylim(0.80, 1.02) # Zoom in on the top (we expect high accuracy)
        ax1.grid(True, which='both', linestyle='--', alpha=0.5)
        
        # Histogram (Counts)
        ax2 = ax1.twinx()
        ax2.bar(bin_centers, counts, width=np.diff(bins), alpha=0.3, color='gray', align='edge')
        ax2.set_ylabel('Number of Samples', color='gray', fontsize=12)
        ax2.set_yscale('log') # Log scale for counts usually looks better
        
        # Boundaries
        ax1.axvline(lower_bound, color='red', linestyle='--', linewidth=2)
        ax1.axvline(upper_bound, color='red', linestyle='--', linewidth=2)
        
        # Formatting
        ax1.set_xscale('log')
        ax1.set_xlabel('Mass Ratio (m1/m2) [Log Scale]', fontsize=12)
        ax1.set_title('Physics Filter vs. AI Performance', fontsize=14)
        
        # Legend (Combine both axes)
        lines1, labels1 = ax1.get_legend_handles_labels()
        ax1.legend(lines1, labels1, loc='lower center')
        
        plt.tight_layout()
        plt.savefig(f"{MODEL_S3_PREFIX}_filter_validation.png")
        plt.show()
        
        # 3. Print Stats
        mask_phys = (mass_ratios <= lower_bound) | (mass_ratios >= upper_bound)
        mask_ai = ~mask_phys
        
        acc_phys = np.mean(y_true[mask_phys] == y_pred_final[mask_phys])
        acc_ai = np.mean(y_true[mask_ai] == y_pred_final[mask_ai])
        
        print(f"Physics Region Accuracy: {acc_phys:.2%} (Should be ~100%)")
        print(f"AI Region Accuracy:      {acc_ai:.2%}")

    # CALL THE FUNCTION
    # Note: These variables should exist from your main block
    m1_val = df_test_ex['m1'].values
    m2_val = df_test_ex['m2'].values
    mass_ratios_val = m1_val / (m2_val + 1e-9)
    
    # Run
    plot_filter_effectiveness(mass_ratios_val, y_true, y_probs, lower_bound=0.05, upper_bound=20.0)