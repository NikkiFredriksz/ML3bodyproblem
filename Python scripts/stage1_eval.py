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
from sklearn.metrics import confusion_matrix, roc_curve, auc

# ==========================================
# CONFIGURATION
# ==========================================
# INPUTS
STORAGE_DB = "sqlite:///three_body_stage_1_v4.db" 
TRAIN_FILE = "train3body.dat"
TEST_FILE = "test3body.dat"

# OUTPUTS
SAVE_MODEL_FILE = "stage1_ion_v23.pth"
SAVE_DATA_FILE = "data_for_stage2.csv"

# SETTINGS (Aggressive Ionization Filtering)
# N_TRIALS: Set to 50 to find best params, or 0 to use defaults/saved
N_TRIALS = 0           
EPOCHS_OPT = 50         
EPOCHS_IONIZATION = 30  
WEIGHT_IONIZATION = 1.0 # High penalty for missing an Ionization
THRESH_IONIZATION = 0.6 # Very low threshold to catch ALL ionizations

# ==========================================
# 1. PHYSICS ENGINE (V22 - Energy Optimized)
# ==========================================
class ThreeBodyPhysics:
    def __init__(self): 
        self.G = 4.302e-3 

    def _calculate_core_physics(self, df):
        """Internal helper to calculate shared physics variables once."""
        m1 = df['m1'].values; m2 = df['m2'].values; m3 = df['m3'].values
        a = df['a_pc'].values; v_inf = df['v_km_s'].values
        
        # Energy Calculation
        E_bin = -self.G * m1 * m2 / (2 * a)
        E_inf = 0.5 * m3 * v_inf**2
        E_tot = E_bin + E_inf
        
        return m1, m2, m3, a, v_inf, E_bin, E_inf, E_tot

    def get_physics_flags(self, df):
        """
        Returns boolean masks and physics values for filtering.
        """
        # Unpack the core physics values calculated above
        _, _, _, _, _, _, _, E_tot = self._calculate_core_physics(df)
        
        # 1. Ionization Possibility (E_tot >= 0)
        # If Total Energy is negative, the system cannot break apart. Ionization is impossible.
        is_possible = E_tot >= 0.0
        
        return is_possible
    
    def convert_batch_to_state(self, df, align=False):
        # Use the internal helper to get vars (avoids copy-paste errors)
        m1, m2, m3, a, v_inf, E_bin, E_inf, E_tot = self._calculate_core_physics(df)
        
        hardness_ratio = E_inf / (np.abs(E_bin) + 1e-9)
        
        # 1. Unpack Variables
        e = df['e'].values; b = df['b_pc'].values
        
        # Keep angles raw
        phi = np.where(np.abs(df['phi'].values)>2*np.pi, np.radians(df['phi'].values), df['phi'].values)
        theta = np.where(np.abs(df['theta'].values)>2*np.pi, np.radians(df['theta'].values), df['theta'].values)
        psi = np.where(np.abs(df['psi'].values)>2*np.pi, np.radians(df['psi'].values), df['psi'].values)
        f = df['f'].values; t_coal = df['t_coal_yr'].values
        
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
        r_mag = (a * (1 - e**2)) / (1 + e * np.cos(f))
        term_h = self.G * M_bin * a * (1 - e**2)
        h_spec = np.sqrt(np.maximum(0.0, term_h))
        inv_h = np.zeros_like(h_spec); mask_h = h_spec > 0
        inv_h[mask_h] = 1.0 / h_spec[mask_h]
        
        vr = (self.G * M_bin * e * np.sin(f)) * inv_h
        vt = h_spec / (r_mag + 1e-9)
        
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
        
        r3 = np.stack([50*a, b, np.zeros_like(a)], axis=1)
        v3 = np.stack([-v_inf, np.zeros_like(v_inf), np.zeros_like(v_inf)], axis=1)

        # --- ANGULAR MOMENTUM CALCS ---
        L_bin_vec = np.cross(r_rel, v_rel)
        L_outer_vec = np.cross(r3, v3)
        
        # NEW: Total Angular Momentum Magnitude
        L_tot_vec = L_bin_vec + L_outer_vec
        L_tot_mag = np.linalg.norm(L_tot_vec, axis=1)

        # Inclination
        dot_L = np.sum(L_bin_vec * L_outer_vec, axis=1)
        norm_L = np.linalg.norm(L_bin_vec, axis=1) * np.linalg.norm(L_outer_vec, axis=1)
        cos_inclination = dot_L / (norm_L + 1e-9)

        # Momentum Ratios
        mu_bin = m1 * m2 / (M_bin + 1e-9)
        L_bin_mag = mu_bin * np.sqrt(self.G * M_bin * a * (1 - e**2) + 1e-9)
        mu_out = m3 * M_bin / (M_tot + 1e-9)
        L_inf_mag = mu_out * b * (v_inf + 1e-9)
        L_ratio = L_inf_mag / (L_bin_mag + 1e-9)

        def lm(x): return np.sign(x)*np.log10(1+np.abs(x))
        r_peri = a * (1 - e)
        compactness = M_tot / (r_peri * (v_inf**2 + 1e-6) + 1e-9)
        
        rm2 = (m2/M_bin)[:,None]; rm1 = (m1/M_bin)[:,None]
        r1 = -rm2*r_rel; r2 = rm1*r_rel
        d13 = np.linalg.norm(r1 - r3, axis=1)
        d23 = np.linalg.norm(r2 - r3, axis=1)
        
        feat = [
            np.log10(m1)[:,None], np.log10(m2)[:,None], np.log10(m3)[:,None],
            np.log10(a)[:,None], 
            np.log10(np.maximum(1e-9, t_coal))[:,None],
            (m1/m2)[:,None], (m2/m3)[:,None], (m3/m1)[:,None],
            
            # THE IONIZATION PREDICTORS
            lm(E_tot)[:,None],                 # Total Energy (Existing)
            np.log10(hardness_ratio)[:,None],  
            
            np.log10(L_ratio)[:,None],
            np.log10(L_tot_mag + 1e-9)[:,None], # Total Angular Momentum (NEW)
            
            np.log10(r_peri + 1e-9)[:,None],
            np.log10((m3/M_bin)*(a/(b+1e-9))**3+1e-9)[:,None], 
            np.sin(f)[:,None], np.cos(f)[:,None],
            feat_phase_sin[:,None], feat_phase_cos[:,None],
            lm(d13-d23)[:, None],
            cos_inclination[:, None],
            np.log10(compactness + 1e-9)[:,None]
        ]
        return np.hstack(feat).astype(np.float32)

# ==========================================
# 2. DATASET (Focus on Chaotic Core)
# ==========================================
class ThreeBodyDataset(Dataset):
    def __init__(self, filepath, physics_engine, mode='ionization', scaler=None, augment=True):
        if not os.path.exists(filepath): sys.exit(f"Error: {filepath} not found.")
        
        # 1. Load Data
        data = pd.read_csv(filepath, sep=r'\s+', engine='python')
        
        # 2. PHYSICS PRE-FILTER (Standard)
        is_possible = physics_engine.get_physics_flags(data)
        data = data[is_possible].copy()
        
        # ---------------------------------------------------------
        # 3. CHAOTIC CORE FILTERING (PHYSICALLY RIGOROUS)
        # ---------------------------------------------------------
        # Calculate the True Closest Approach (r_min) accounting for Gravitational Focusing.
        # Formula: r_min derived from Hyperbolic Orbit Mechanics
        
        G = 4.302e-3
        M_tot = data['m1'] + data['m2'] + data['m3']
        v_inf = data['v_km_s']
        b = data['b_pc']
        
        # Calculate Hyperbolic Semi-Major Axis (a_hyp)
        # Note: Adding 1e-9 to v_inf to avoid division by zero
        a_hyp = (G * M_tot) / (v_inf**2 + 1e-9)
        
        # Calculate Eccentricity of the encounter orbit
        e_hyp = np.sqrt(1 + (b / a_hyp)**2)
        
        # True Closest Approach Distance
        r_min = a_hyp * (e_hyp - 1)
        
        # Logic: If the star actually gets within 5 binary radii, it's dangerous.
        is_flyby = r_min > (5.0 * data['a_pc'])
        
        # ---------------------------------------------------------
        # Keep (Close Encounters) OR (Actual Ionizations)
        mask_chaotic = (~is_flyby) | (data['OUTCOME'] == 3)
        
        n_dropped = len(data) - np.sum(mask_chaotic)
        data = data[mask_chaotic].copy()
        
        print(f"[Dataset] Chaotic Core: Dropped {n_dropped} safe fly-bys (Calculated via r_min).")
        
        # ---------------------------------------------------------
        
        # 4. Get Labels
        raw_outcomes = data['OUTCOME'].astype(int).values
        y_raw = (raw_outcomes == 3).astype(int)
        
        # 5. Oversampling (Standard)
        dfs = [data]
        ys = [y_raw]
        
        if augment and mode == 'ionization':
            mask_ion = (y_raw == 1)
            count_ion = np.sum(mask_ion)
            count_bound = len(y_raw) - count_ion
            
            if count_ion > 0:
                multiplier = int(count_bound / count_ion) - 1
                if multiplier > 0:
                    df_ion = data[mask_ion].copy()
                    y_ion = y_raw[mask_ion].copy()
                    for _ in range(multiplier):
                        dfs.append(df_ion)
                        ys.append(y_ion)
                    print(f"[Oversampling] Balanced with {multiplier}x duplication.")

        self.df_final = pd.concat(dfs, ignore_index=True)
        self.y = np.concatenate(ys)
        print(f"[Dataset] Final Size: {len(self.y)} (Focusing on Hard Cases)")

        # 6. Generate States
        self.X_orig = physics_engine.convert_batch_to_state(self.df_final)
        
        # Generate Mirror State
        df_mirror = self.df_final.copy()
        df_mirror['m1'], df_mirror['m2'] = self.df_final['m2'], self.df_final['m1']
        df_mirror['psi'] += np.pi 
        self.X_mirror = physics_engine.convert_batch_to_state(df_mirror)
        
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
    Ionization potential doesn't change if we swap star 1 and star 2.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
    def forward(self, x, x_mirror):
        return (self.base(x) + self.base(x_mirror)) / 2.0

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
 
# ==========================================
# 5. MAIN (EVALUATION ONLY)
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    physics = ThreeBodyPhysics()
    
    # 1. SETUP DATA (Needed for Scaler)
    print("Loading Training Data (to recover Scaler)...")
    # We load this ONLY to fit the scaler so test data matches training normalization
    ds_train = ThreeBodyDataset(TRAIN_FILE, physics, mode='ionization', augment=True)
    
    # 2. LOAD SAVED MODEL
    print(f"Loading Model from {SAVE_MODEL_FILE}...")
    
    # [IMPORTANT] Set these to match the 'Best Params' from your training run!
    # If your saved model has 256 neurons and 4 layers, these must match.
    H_DIM = 512
    N_LAYERS = 4
    
    # Initialize the architecture (Must match the saved file)
    # We use ds_train.input_dim to ensure input size is correct
    sample = ds_train[0]
    first_sample_x = sample[0] # The input tensor is always the first item
    input_dim_size = first_sample_x.shape[0]

    # [FIX] 1. Build the Base Brain first
    base_network = ThreeBodyResNet(input_dim=input_dim_size, hidden_dim=H_DIM, num_layers=N_LAYERS)
    
    # [FIX] 2. Wrap it in the Invariance Shell
    model = InvariantThreeBodyNet(base_network).to(device)
    
    if os.path.exists(SAVE_MODEL_FILE):
        model.load_state_dict(torch.load(SAVE_MODEL_FILE, map_location=device))
        print("Model loaded successfully.")
    else:
        sys.exit(f"Error: {SAVE_MODEL_FILE} not found!")

    # 3. EVALUATE (SMART INFERENCE)
    print(f"\nEvaluating on {TEST_FILE}...")
    df_test = pd.read_csv(TEST_FILE, sep=r'\s+', engine='python')
    
    # Note: Adjust this target mapping if your data uses different outcome codes
    true_labels = (df_test['OUTCOME'].astype(int) == 3).astype(int).values 
    
    # Vectorized Physics Conversion
    # We use the scaler from ds_train to ensure consistent normalization
    X_test_orig = torch.tensor(ds_train.scaler.transform(physics.convert_batch_to_state(df_test)), dtype=torch.float32).to(device)
    
    # Mirror view for Invariant Network
    df_mirror = df_test.copy()
    df_mirror['m1'], df_mirror['m2'] = df_test['m2'], df_test['m1']; df_mirror['psi'] += np.pi
    X_test_mirror = torch.tensor(ds_train.scaler.transform(physics.convert_batch_to_state(df_mirror)), dtype=torch.float32).to(device)
    
    model.eval()
    with torch.no_grad():
        # Get raw probabilities (Handling the Siamese/Invariant Inputs)
        # We output Softmax prob for Class 1 (Ionization)
        probs = torch.nn.functional.softmax(model(X_test_orig, X_test_mirror), dim=1)[:, 1].cpu().numpy()
        
    # --- SMART PREDICTION LOGIC (HYBRID PIPELINE - V2 PHYSICS) ---
    print("Applying Physics Veto (Energy & Gravitational Focusing)...")
    
    # 1. Get Physics Flags
    is_possible = physics.get_physics_flags(df_test)
    
    # 2. Calculate "Safe Fly-by" Filter using Hyperbolic Mechanics
    # We calculate r_min (True Closest Approach) accounting for Gravitational Focusing.
    # If the star is slow, gravity pulls it closer than 'b' implies.
    
    G = 4.302e-3
    M_tot = df_test['m1'] + df_test['m2'] + df_test['m3']
    v_inf = df_test['v_km_s']
    b = df_test['b_pc']
    
    # Calculate Hyperbolic Semi-Major Axis (a_hyp)
    # (Avoid division by zero by adding epsilon)
    a_hyp = (G * M_tot) / (v_inf**2 + 1e-9)
    
    # Calculate Eccentricity of the encounter orbit
    e_hyp = np.sqrt(1 + (b / a_hyp)**2)
    
    # Calculate True Pericenter (r_min)
    r_min = a_hyp * (e_hyp - 1)
    
    # Logic: If the star actually penetrates within 5 binary radii, it's a "Close Encounter".
    # Otherwise, it's a "Safe Fly-by".
    is_flyby = r_min > (5.0 * df_test['a_pc'])
    
    # 3. Initialize Predictions to 0 (Bound)
    preds = np.zeros(len(df_test), dtype=int)
    
    # 4. Apply Logic
    
    # Define the "AI Zone" (The Chaos Core):
    # 1. Must be Physically Possible (E_tot >= 0)
    # 2. Must NOT be a safe fly-by (r_min < 5a)
    mask_ai_zone = is_possible & (~is_flyby)
    
    # Apply AI predictions ONLY in the "Hard" AI Zone
    preds[mask_ai_zone] = (probs[mask_ai_zone] > THRESH_IONIZATION).astype(int)
    
    # Reporting
    n_imp = np.sum(~is_possible)
    n_fly = np.sum(is_flyby & is_possible)
    n_ai = np.sum(mask_ai_zone)
    print(f"   - Physically Impossible (Veto): {n_imp} samples")
    print(f"   - Safe Fly-bys (r_min > 5a):    {n_fly} samples")
    print(f"   - Chaotic Core (AI Decides):    {n_ai} samples")
    
    # ----------------------------------------
    
    # 4. FILTER & SAVE DATA
    # Filter for Stage 2 (Keep Non-Ionization, i.e., Pred == 0)
    indices_for_stage2 = [i for i, p in enumerate(preds) if p == 0]
    
    print(f"Total Test Samples: {len(df_test)}")
    print(f"Classified as Ionization: {np.sum(preds == 1)}")
    print(f"Passed to Stage 2 (Non-Ionization): {len(indices_for_stage2)}")
    
    if len(indices_for_stage2) > 0:
        df_stage2 = df_test.iloc[indices_for_stage2].copy()
        df_stage2.to_csv(SAVE_DATA_FILE, sep=' ', index=False)
        print(f"Saved filtered data to '{SAVE_DATA_FILE}'")
    else:
        print("Warning: All samples were classified as Ionized. Stage 2 file is empty.")

    # ==========================================
    # 4. DIAGNOSTICS & PLOTS
    # ==========================================
    print("\n--- Running Advanced Physics Diagnostics ---")
    
    # We recalculate these ONLY for plotting purposes (safe to do here)
    m1_test = df_test['m1'].values; m2_test = df_test['m2'].values; m3_test = df_test['m3'].values
    a_test = df_test['a_pc'].values; v_inf_test = df_test['v_km_s'].values
    G = 4.302e-3
    E_bin = -G * m1_test * m2_test / (2 * a_test)
    E_inf = 0.5 * m3_test * v_inf_test**2
    hardness_ratio_val = E_inf / (np.abs(E_bin) + 1e-9)
    mass_ratio_val = m1_test / (m2_test + 1e-9)

    # 1. CONFIDENCE HISTOGRAM
    plt.figure(figsize=(7,4))
    plt.hist(probs, bins=50, alpha=0.7, color='orange')
    plt.title("Distribution of Ionization Probabilities")
    plt.xlabel("Probability (0 = Bound, 1 = Ionization)")
    plt.ylabel("Count")
    plt.axvline(THRESH_IONIZATION, color='red', linestyle='--', label=f'Threshold ({THRESH_IONIZATION})')
    plt.legend()
    plt.savefig("stage1_histogram.png")
    plt.show()

    # 2. PHYSICS ACCURACY PLOTS
    def plot_binned_accuracy(variable, var_name, y_true, y_pred, bins=10, log_scale=False):
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
        
        plt.savefig(f"stage1_acc_vs_{var_name.split()[0]}.png")
        plt.show()

    # Plot 1: Accuracy vs Hardness
    plot_binned_accuracy(hardness_ratio_val, "Hardness Ratio (E_kin / E_bin)", true_labels, preds, log_scale=True)
    
    # Plot 2: Accuracy vs Mass Ratio
    plot_binned_accuracy(mass_ratio_val, "Mass Ratio (m1/m2)", true_labels, preds, log_scale=False)

    # 3. FAILURE MAP
    plt.figure(figsize=(8, 6))
    correct_mask = (true_labels == preds)
    log_hardness = np.log10(hardness_ratio_val + 1e-9)
    
    plt.scatter(mass_ratio_val[correct_mask], log_hardness[correct_mask], 
                c='green', s=5, alpha=0.1, label='Correct')
    plt.scatter(mass_ratio_val[~correct_mask], log_hardness[~correct_mask], 
                c='red', marker='x', s=20, alpha=0.6, label='Wrong')
    
    plt.title("Failure Map: Mass Ratio vs Hardness")
    plt.xlabel("Mass Ratio (m1/m2)")
    plt.ylabel("Log10 Hardness Ratio")
    plt.legend()
    plt.savefig("stage1_failure_map.png")
    plt.show()

    # 4. CONFUSION MATRIX
    cm = confusion_matrix(true_labels, preds)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=['Bound', 'Ionized'], yticklabels=['Bound', 'Ionized'])
    plt.title('Stage 1: Ionization Recall')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig("stage1_confusion_matrix.png")
    plt.show()
    
    # 4a. RAW CONFUSION MATRIX (COUNTS)
    # Calculate matrix
    cm = confusion_matrix(true_labels, preds)

    # Print raw numbers to console
    print("\nConfusion Matrix (Raw Counts):")
    print(cm)
    
    # Plot Raw Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Bound (0)', 'Ionized (1)'], 
                yticklabels=['Bound (0)', 'Ionized (1)'])
    plt.title('Stage 1: Confusion Matrix (Raw Counts)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig("stage1_confusion_matrix_raw.png")
    plt.show()


    probs_clean = probs.copy()
    probs_clean[~is_possible] = 0.0
    # 6. ROC CURVE
    fpr, tpr, thresholds = roc_curve(true_labels, probs_clean)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig("stage1_roc.png")
    plt.show()
    
    print(f"\nFinal Ionization Recall: {cm[1,1] / (cm[1,1] + cm[1,0]):.2%}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # ==========================================
    # 7. PHYSICS FILTER VALIDATION PLOT (UPDATED)
    # ==========================================
    print("\n--- Generating Physics Filter Validation Plot ---")

    # 1. Define the 3 Zones of your Hybrid Model
    
    # Zone A: Impossible (Energy < 0)
    # Logic: Physics says "Bound" (0). We check if that is correct.
    mask_impossible = ~is_possible
    
    # Zone B: Safe Fly-bys (r_min > 5a)
    # Logic: We use the rigorous 'is_flyby' flag calculated via Hyperbolic Mechanics.
    # We recalculate it here to ensure the plotting logic is self-contained and verifiable.
    mask_flyby = is_flyby & is_possible
    
    # Zone C: Chaotic Core (The Rest)
    # Logic: This is where the AI actually predicts.
    mask_ai = is_possible & (~mask_flyby)
    
    # 2. Calculate Accuracy for each Zone
    # For Zone A & B, "Prediction" is always 0. We compare 0 to True Label.
    
    # Acc Impossible
    acc_imp = np.mean(true_labels[mask_impossible] == 0) if np.sum(mask_impossible) > 0 else 0.0
    
    # Acc Fly-by (CRITICAL CHECK: Is the filter too strong?)
    # If this is < 1.0, it means there are Ionizations in the Fly-by zone!
    acc_fly = np.mean(true_labels[mask_flyby] == 0) if np.sum(mask_flyby) > 0 else 0.0
    
    # Acc AI (How smart is the Neural Net?)
    acc_ai = np.mean(true_labels[mask_ai] == preds[mask_ai]) if np.sum(mask_ai) > 0 else 0.0
    
    # 3. Prepare Data
    accuracies = [acc_imp, acc_fly, acc_ai]
    counts = [np.sum(mask_impossible), np.sum(mask_flyby), np.sum(mask_ai)]
    labels = ['Impossible\n(Energy Veto)', 'Fly-bys\n(Distance Veto)', 'Chaotic Core\n(AI Prediction)']
    colors = ['green', 'orange', 'blue'] # Green=Physics, Blue=AI
    
    # 4. Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels, accuracies, color=colors, alpha=0.7)
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title('Hybrid Pipeline Performance\nFly-by Filter = 5.0a', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add labels
    for bar, acc, count in zip(bars, accuracies, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{acc:.2%}\n(N={count})',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
                 
    plt.tight_layout()
    plt.savefig("stage1_filter_validation.png")
    plt.show()
    
    # ==========================================
    # 8. IMPACT ANALYSIS: PURE AI vs HYBRID
    # ==========================================
    print("\n--- Generating Impact Analysis Plot ---")
    
    # 1. Calculate Pure AI Accuracy (ignoring physics entirely)
    # We use the same threshold you set for the hybrid model
    preds_pure_ai = (probs > THRESH_IONIZATION).astype(int)
    acc_pure_ai = np.mean(true_labels == preds_pure_ai)
    
    # 2. Hybrid Accuracy (AI + Physics Veto)
    # 'preds' is already the hybrid prediction calculated in your Main block
    acc_hybrid = np.mean(true_labels == preds)
    
    print(f"Pure AI Accuracy: {acc_pure_ai:.2%}")
    print(f"Hybrid Accuracy:  {acc_hybrid:.2%}")
    print(f"Improvement:      {acc_hybrid - acc_pure_ai:+.2%}")

    # 3. Plot Comparison
    labels = ['Pure AI', 'Hybrid\n(AI + Physics)']
    values = [acc_pure_ai, acc_hybrid]
    colors = ['gray', 'green']
    
    plt.figure(figsize=(6, 6))
    bars = plt.bar(labels, values, color=colors, alpha=0.7, width=0.6)
    
    plt.ylim(0, 1.1)
    plt.ylabel('Global Accuracy')
    plt.title(f'Impact of Physics Filter\n(Threshold = {THRESH_IONIZATION})')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add numbers on top
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, v + 0.02, 
                 f"{v:.2%}", ha='center', fontweight='bold')
                 
    plt.tight_layout()
    plt.savefig("stage1_impact_analysis.png")
    plt.show()