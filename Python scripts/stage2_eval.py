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
import torch.nn.functional as F

# ==========================================
# CONFIGURATION
# ==========================================
# INPUTS
STORAGE_DB = "sqlite:///three_body_stage_2_v4.db"  # Database URL for storing Optuna trials
TRAIN_FILE = "train3body.dat"     # Input file containing training simulation data
TEST_FILE = "data_for_stage2.csv" # Input file containing test data (output from Stage 1)

# OUTPUTS
SAVE_MODEL_FILE = "stage2_int_v22.pth"  # Filename to save the trained model weights
SAVE_DATA_FILE = "data_for_stage3.csv"  # Filename to save data filtered as 'Exchange' for the next stage

# SETTINGS
N_MODELS_S2 = 3           # Number of models to train for the ensemble (improves robustness)
N_TRIALS = 45             # Number of Optuna trials for hyperparameter tuning
EPOCHS_OPT = 50           # Number of epochs per trial during optimization
EPOCHS_INTERACTION = 300  # Number of epochs for the final full training run
WEIGHT_INTERACTION = 5.0  # Class weight for the 'Flyby' class to handle class imbalance
THRESH_INTERACTION = 0.70 # Probability threshold for classifying an interaction (Exchange)

# ==========================================
# 1. PHYSICS ENGINE
# ==========================================
class ThreeBodyPhysics:
    """
    Handles physics-based feature engineering.
    
    This class transforms raw initial conditions (orbital elements) into physically 
    meaningful features (energies, momenta, relative coordinates) that help the 
    Neural Network learn the dynamics of the 3-body problem effectively.
    """
    def __init__(self): 
        self.G = 4.302e-3  # Gravitational constant in specific simulation units (pc * (km/s)^2 / M_sun)

    def convert_batch_to_state(self, df):
        """
        Converts a raw Pandas DataFrame batch into a processed feature tensor.
        
        Raw orbital elements (like angles) are often cyclic or poorly scaled for NNs.
        This function computes invariant physical quantities (Energy, Angular Momentum)
        and projects the system into a consistent 3D coordinate frame to make the 
        learning task easier for the model.
        """
        # 1. Unpack Variables
        m1 = df['m1'].values; m2 = df['m2'].values; m3 = df['m3'].values # Masses of the three bodies
        a = df['a_pc'].values; e = df['e'].values; b = df['b_pc'].values # Semi-major axis, eccentricity, impact parameter
        
        # Keep angles raw but normalize them to be within reasonable bounds for rotation logic
        # np.where checks if angle > 2*pi, if so converts to radians (assuming mixed units), else keeps as is
        phi = np.where(np.abs(df['phi'].values)>2*np.pi, np.radians(df['phi'].values), df['phi'].values)
        theta = np.where(np.abs(df['theta'].values)>2*np.pi, np.radians(df['theta'].values), df['theta'].values)
        psi = np.where(np.abs(df['psi'].values)>2*np.pi, np.radians(df['psi'].values), df['psi'].values)
        
        f = df['f'].values; v_inf = df['v_km_s'].values; t_coal = df['t_coal_yr'].values # True anomaly, velocity at infinity, coalescence time
        
        # --- FEATURE ENGINEERING ---
        M_bin = m1 + m2          # Total mass of the inner binary
        M_tot = m1 + m2 + m3     # Total mass of the entire 3-body system
        
        # --- PHASE FEATURES ---
        r_peri_encounter = b     # Periapsis distance of the encounter (approximation using impact parameter b)
        
        # Calculate velocity at periapsis using energy conservation (vis-viva equation approximation)
        v_peri_encounter = np.sqrt(v_inf**2 + 2*self.G*M_bin/(r_peri_encounter+1e-9))
        
        v_avg = np.sqrt(v_inf * v_peri_encounter)  # Geometric mean velocity for time estimation
        t_approach = (50.0 * a) / (v_avg + 1e-9)   # Estimated time to approach the binary (from 50*a distance)

        mean_motion = np.sqrt(self.G * M_bin / (a**3 + 1e-9)) # Angular speed of the binary orbit
        M_encounter = f + mean_motion * t_approach            # Mean anomaly at the time of encounter
        
        feat_phase_sin = np.sin(M_encounter)  # Sine of encounter phase (handles cyclic nature)
        feat_phase_cos = np.cos(M_encounter)  # Cosine of encounter phase
        
        # --- COORDINATE TRANSFORMS ---
        # 1. Calculate Magnitude (r_mag)
        # Distance between binary stars based on ellipse equation
        r_mag = (a * (1 - e**2)) / (1 + e * np.cos(f))
        
        # 2. Calculate Angular Momentum (h) and Velocity Components
        term_h = self.G * M_bin * a * (1 - e**2)      # Squared specific angular momentum
        h_spec = np.sqrt(np.maximum(0.0, term_h))     # Specific angular momentum (handle negatives due to precision)
        
        inv_h = np.zeros_like(h_spec); mask_h = h_spec > 0 # Safe inverse calculation mask
        inv_h[mask_h] = 1.0 / h_spec[mask_h]               # Calculate 1/h where h > 0
        
        # Radial velocity component (vr) and Tangential velocity component (vt)
        vr = (self.G * M_bin * e * np.sin(f)) * inv_h
        vt = h_spec / (r_mag + 1e-9) 
        
        # 3. Project to Plane (2D relative coordinates of binary)
        c_f, s_f = np.cos(f), np.sin(f)
        r_rel_plane = np.stack([r_mag * c_f, r_mag * s_f, np.zeros_like(f)], axis=1)          # Position vector in orbital plane
        v_rel_plane = np.stack([vr * c_f - vt * s_f, vr * s_f + vt * c_f, np.zeros_like(f)], axis=1) # Velocity vector in orbital plane
        
        # 4. Rotate to 3D Space (Apply Euler rotations: phi, theta, psi)
        z = np.zeros_like(phi); o = np.ones_like(phi) # Zero and One arrays for matrix construction
        
        # Rotation Matrix Z (phi)
        c, s = np.cos(phi), np.sin(phi)
        Rz_phi = np.stack([np.stack([c,-s,z],1), np.stack([s,c,z],1), np.stack([z,z,o],1)],1)
        
        # Rotation Matrix X (theta)
        c, s = np.cos(theta), np.sin(theta)
        Rx_theta = np.stack([np.stack([o,z,z],1), np.stack([z,c,-s],1), np.stack([z,s,c],1)],1)
        
        # Rotation Matrix Z (psi)
        c, s = np.cos(psi), np.sin(psi)
        Rz_psi = np.stack([np.stack([c,-s,z],1), np.stack([s,c,z],1), np.stack([z,z,o],1)],1)
        
        # Combined Rotation Matrix
        R = Rz_phi @ Rx_theta @ Rz_psi
        
        # Apply rotation to get 3D relative position and velocity vectors
        r_rel = (R @ r_rel_plane[:,:,None]).squeeze(-1)
        v_rel = (R @ v_rel_plane[:,:,None]).squeeze(-1)
        
        # Define incoming intruder (body 3) vector relative to center of mass
        r3 = np.stack([50*a, b, np.zeros_like(a)], axis=1)               # Intruder starts at 50*a
        v3 = np.stack([-v_inf, np.zeros_like(v_inf), np.zeros_like(v_inf)], axis=1) # Incoming velocity along x-axis

        # --- CALCULATE INCLINATION ---
        L_bin_vec = np.cross(r_rel, v_rel)   # Angular momentum vector of binary
        L_outer_vec = np.cross(r3, v3)       # Angular momentum vector of the encounter
        
        # Dot product and magnitudes to find cosine of inclination angle
        dot_L = np.sum(L_bin_vec * L_outer_vec, axis=1)
        norm_L = np.linalg.norm(L_bin_vec, axis=1) * np.linalg.norm(L_outer_vec, axis=1)
        cos_inclination = dot_L / (norm_L + 1e-9) # Cosine(i), ranges -1 to 1
        
        # --- PHYSICS 1: ENERGIES ---
        E_bin = -self.G * m1 * m2 / (2 * a)    # Binding energy of the binary
        E_inf = 0.5 * m3 * v_inf**2            # Kinetic energy of the intruder at infinity
        E_tot = E_bin + E_inf                  # Total energy of the system
        hardness_ratio = E_inf / (np.abs(E_bin) + 1e-9) # Ratio of kinetic to binding energy (Key stability metric)

        # --- PHYSICS 2: MOMENTUM MAGNITUDES ---
        mu_bin = m1 * m2 / (M_bin + 1e-9)      # Reduced mass of binary
        L_bin_mag = mu_bin * np.sqrt(self.G * M_bin * a * (1 - e**2) + 1e-9) # Magnitude of binary angular momentum
        
        mu_out = m3 * M_bin / (M_tot + 1e-9)   # Reduced mass of outer system
        L_inf_mag = mu_out * b * (v_inf + 1e-9)# Magnitude of encounter angular momentum
        L_ratio = L_inf_mag / (L_bin_mag + 1e-9) # Ratio of angular momenta

        # Helpers
        def lm(x): return np.sign(x)*np.log10(1+np.abs(x)) # Log-modulus transformation for scaling large values
        
        r_peri = a * (1 - e)   # Periapsis distance of binary
        # Compactness parameter: relates density of the system to interaction speed
        compactness = M_tot / (r_peri * (v_inf**2 + 1e-6) + 1e-9)
        
        # --- DERIVED GEOMETRY ---
        rm2 = (m2/M_bin)[:,None]; rm1 = (m1/M_bin)[:,None] # Mass fractions
        r1 = -rm2*r_rel; r2 = rm1*r_rel  # Positions of m1 and m2 relative to binary COM
        
        d13 = np.linalg.norm(r1 - r3, axis=1) # Distance between m1 and m3
        d23 = np.linalg.norm(r2 - r3, axis=1) # Distance between m2 and m3
        
        # [NEW] Calculate Mass Ratio Proximity to 1 (The "Chaos Factor")
        # Exchanges are most likely when masses are similar
        q_prox = np.abs(1.0 - (m1/m2))
        
        # --- UPGRADE 2: PERTURBATION STRENGTH FEATURES ---
        # 1. Orbital Velocity of Inner Binary (Measure of binding strength)
        v_orb_bin = np.sqrt(self.G * M_bin / (a + 1e-9))

        # 2. Tidal Kick (Delta V) estimate at closest approach
        # Approximation: Impulse ~ Force * Time
        # Force ~ G*m3 / b^2   |   Time ~ b / v_inf
        # Delta V ~ (G*m3 / b^2) * (b / v_inf) = G*m3 / (b * v_inf)
        delta_v_approx = (2 * self.G * m3) / (b * v_inf + 1e-9)

        # 3. The "Chaos Parameter": Ratio of Kick to Binding Strength
        # If this > 1.0, the tidal kick is stronger than the orbital velocity -> likely destruction/exchange
        perturbation_strength = delta_v_approx / (v_orb_bin + 1e-9)
        
        # [NEW FEATURE] Impact Penetration Depth
        # Low value (< 2.0) = Deep penetration into binary (Chaos)
        # High value (> 5.0) = Distant flyby (Stable)
        penetration_depth = b / (a + 1e-9)
        
        # 5. Final Feature Assembly (Stacking all computed features)
        # Using Log10 on features spanning many orders of magnitude to aid NN convergence
        feat = [
            np.log10(penetration_depth + 1e-9)[:, None],
            np.log10(m1)[:,None], np.log10(m2)[:,None], np.log10(m3)[:,None],
            np.log10(a)[:,None], 
            np.log10(np.maximum(1e-9, t_coal))[:,None],
            (m1/m2)[:,None], (m2/m3)[:,None], (m3/m1)[:,None],
            lm(E_tot)[:,None],                 
            np.log10(hardness_ratio)[:,None],  
            np.log10(L_ratio)[:,None],         
            np.log10(r_peri + 1e-9)[:,None],
            np.log10((m3/M_bin)*(a/(b+1e-9))**3+1e-9)[:,None], 
            np.sin(f)[:,None], np.cos(f)[:,None],
            feat_phase_sin[:,None], feat_phase_cos[:,None],
            lm(d13-d23)[:, None],
            cos_inclination[:, None],
            np.log10(compactness + 1e-9)[:,None],
            np.log10(q_prox + 1e-9)[:, None],
            np.log10(perturbation_strength + 1e-9)[:, None] 
        ]
        # Concatenate list of arrays into a single 2D Numpy array (Batch x Features)
        return np.hstack(feat).astype(np.float32)

# ==========================================
# 2. DATASET
# ==========================================
class ThreeBodyDataset(Dataset):
    """
    Custom PyTorch Dataset for loading and processing 3-body simulation data.
    
    1. It loads data from disk and filters out 'Ionization' cases (Class 3), as Stage 2 
       only distinguishes between 'Flyby' (Class 0) and 'Exchange' (Classes 1 & 2).
    2. It implements 'Symmetry by Design': For every interaction, it generates a 'Mirror State'
       where stars 1 and 2 are swapped. This allows the model to enforce physical symmetry.
    """
    def __init__(self, filepath, physics_engine, mode='interaction', scaler=None, augment=False):
        # check if file exists, exit if not
        if not os.path.exists(filepath): sys.exit(f"Error: {filepath} not found.")

        # Load data with python engine to support regex separators
        data = pd.read_csv(filepath, sep=r'\s+', engine='python')
        
        # 2. Filter for Interaction Mode Labels
        raw_outcomes = data['OUTCOME'].astype(int).values
        
        if mode == 'interaction':
            # Create a mask to remove Ionization (Class 3) events
            # Stage 1 already filtered these, but we double-check or handle raw files here.
            mask = raw_outcomes != 3 
            
            # Apply mask to dataframe and labels
            data = data[mask].copy()
            raw_outcomes = raw_outcomes[mask]
            
            # Define Binary Classification Targets:
            # 0 -> Flyby (Outcome 0)
            # 1 -> Exchange (Outcome 1 or 2)
            self.y = (raw_outcomes > 0).astype(int)
        
        # 3. Generate States (Feature Engineering)
        # Convert the filtered dataframe into the physics-rich feature matrix
        self.X_orig = physics_engine.convert_batch_to_state(data)
        
        # 2. Generate Mirror State (Symmetry)
        # Create a copy of the data where star 1 and star 2 are swapped
        df_mirror = data.copy()
        df_mirror['m1'], df_mirror['m2'] = data['m2'], data['m1'] # Swap masses
        df_mirror['psi'] += np.pi # Rotate the binary phase by 180 degrees (pi radians)
        
        # Generate features for this "mirrored" view of the same system
        self.X_mirror = physics_engine.convert_batch_to_state(df_mirror)
        
        # Note: Standard data augmentation (noise injection) is removed because 
        # the Invariant Model architecture handles generalization mathematically.
        
        # Scaling (Standardization)
        if scaler:
            # If a scaler is provided (e.g., from training set), use it to transform data
            self.X_orig = scaler.transform(self.X_orig)
            self.X_mirror = scaler.transform(self.X_mirror)
            self.scaler = scaler
        else:
            # If training, fit a new Standard Scaler
            self.scaler = StandardScaler()
            
            # Fit on BOTH original and mirror views. 
            # This ensures the scaler respects the physical symmetry of the problem.
            combined = np.concatenate([self.X_orig, self.X_mirror], axis=0)
            self.scaler.fit(combined)
            
            # Transform both sets
            self.X_orig = self.scaler.transform(self.X_orig)
            self.X_mirror = self.scaler.transform(self.X_mirror)
            
    def __len__(self): return len(self.y) # Return total number of samples
    
    def __getitem__(self, idx): 
        # Return a tuple: (Original Features, Swapped Features, Label)
        return (torch.tensor(self.X_orig[idx]), 
                torch.tensor(self.X_mirror[idx]), 
                torch.tensor(self.y[idx], dtype=torch.long))

# ==========================================
# 3. MODEL
# ==========================================
class ResidualBlock(nn.Module):
    """
    A standard Residual Block for Deep Neural Networks.
    
    Deep networks can suffer from vanishing gradients. Residual connections (x + f(x))
    allow gradients to flow through the network easily, enabling to train 
    deeper models (e.g., 6-8 layers) effectively.
    """
    def __init__(self, hidden_dim, dropout_rate):
        super(ResidualBlock, self).__init__()
        # Two linear layers with Batch Norm, GELU activation, and Dropout
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout_rate)
        )
    def forward(self, x): 
        # Add the input (residual) to the output of the block
        return x + self.block(x)

class InvariantThreeBodyNet(nn.Module):
    """
    A wrapper class that enforces Permutation Invariance.
    
    Why this is here:
    In the 3-body problem, swapping Star 1 and Star 2 does not change the *fundamental nature*
    of the outcome classification (Flyby vs Exchange). 
    
    Mathematical Logic:
    P(Exchange | m1, m2) should equal P(Exchange | m2, m1).
    This model computes: Output = ( Network(m1, m2) + Network(m2, m1) ) / 2
    This forces the prediction to be identical regardless of input order.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model # The underlying ResNet
        
    def forward(self, x, x_mirror):
        # Pass the original view through the network
        out_orig = self.base(x)
        # Pass the swapped (mirror) view through the SAME network
        out_mirror = self.base(x_mirror)
        
        # Average the logits. 
        # Unlike Stage 3 (which detects direction 1-3 vs 2-3 and uses subtraction),
        # Stage 2 detects "Any Exchange", so we ADD/Average to enforce symmetry.
        return (out_orig + out_mirror) / 2.0

class ThreeBodyResNet(nn.Module):
    """
    The main feed-forward neural network architecture.
    """
    def __init__(self, input_dim, output_dim=2, hidden_dim=512, num_layers=4, dropout_rate=0.05):
        super(ThreeBodyResNet, self).__init__()
        
        # Input projection layer
        self.input_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU())
        
        # Stack multiple Residual Blocks
        layers = []
        for _ in range(num_layers):
            layers.append(ResidualBlock(hidden_dim, dropout_rate))
        self.res_blocks = nn.Sequential(*layers)
        
        # Output Head: Projects hidden state down to class logits
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), 
            nn.GELU(), 
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x): 
        # Pass through Input -> ResBlocks -> Output Head
        return self.output_head(self.res_blocks(self.input_layer(x)))

# ==========================================
# 5. MAIN
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    physics = ThreeBodyPhysics()
    
    # 1. LOAD TRAINING DATA
    print("Initializing Training Data...")
    ds_train_fit = ThreeBodyDataset(TRAIN_FILE, physics, mode='ionization', augment=False)
    # Note: augment=False because Invariant Model handles it
    ds_train = ThreeBodyDataset(TRAIN_FILE, physics, mode='interaction', scaler=ds_train_fit.scaler, augment=False)
    
    # ==========================================
    # STEP 3: LOAD MODEL ENSEMBLE (3 BRAINS)
    # ==========================================
    print("Loading Ensemble of 3 Models...")
    
    # Hyperparameters from training
    H_DIM = 512 
    N_LAYERS = 5
    DROPOUT = 0.05
    
    models = []
    
    # Load Brains 0, 1, and 2
    for i in range(3):
        fname = f"stage2_model_{i}.pth"
        
        if os.path.exists(fname):
            # Initialize fresh architecture
            base = ThreeBodyResNet(ds_train.X_orig.shape[1], 2, H_DIM, N_LAYERS, DROPOUT)
            model = InvariantThreeBodyNet(base).to(device)
            
            # [FIX IS HERE]: Add map_location=device
            checkpoint = torch.load(fname, map_location=device)
            model.load_state_dict(checkpoint)
            
            model.eval() # Freeze for inference
            models.append(model)
            print(f"   > Loaded {fname}")
        else:
            print(f"   ! WARNING: {fname} not found. Skipping.")
            
    if len(models) == 0:
        print("CRITICAL: No models loaded!")
        sys.exit()
    
    # 4. EVALUATE ON FILTERED DATA (HYBRID MODE)
    if not os.path.exists(TEST_FILE):
        print(f"CRITICAL: '{TEST_FILE}' not found. Run Stage 1 script first.")
        sys.exit()
        
    print(f"\nEvaluating on {TEST_FILE} (Filtered from Stage 1)...")
    df_test = pd.read_csv(TEST_FILE, sep=r'\s+', engine='python')
    true_labels = (df_test['OUTCOME'].astype(int) > 0).astype(int).values
    
    # --- [UPGRADE] HYBRID INFERENCE LOGIC (CORRECTED) ---
    print("Analytic Stability Filter DISABLED for Inference. Trusting Neural Network...")
    
    # 1. Prepare Tensors for THE WHOLE DATASET (No filtering)
    X_test_orig = torch.tensor(ds_train.scaler.transform(physics.convert_batch_to_state(df_test)), dtype=torch.float32).to(device)
    
    df_mirror = df_test.copy()
    df_mirror['m1'], df_mirror['m2'] = df_test['m2'], df_test['m1']; df_mirror['psi'] += np.pi
    X_test_mirror = torch.tensor(ds_train.scaler.transform(physics.convert_batch_to_state(df_mirror)), dtype=torch.float32).to(device)
    
    # --- [FIX START] ENSEMBLE PREDICTIONS ---
    print(f"Running Ensemble Inference with {N_MODELS_S2} Models...")
    
    avg_probs = np.zeros(len(df_test))
    
    with torch.no_grad():
        # [FIX] Loop over the list of 'models' we just got
        for i, model in enumerate(models):
            model.eval()
            # 1. Forward Pass
            logits = model(X_test_orig, X_test_mirror)
            # 2. Convert Logits to Probabilities (Softmax)
            probabilities = F.softmax(logits, dim=1)
            # 3. Add to average
            avg_probs += probabilities[:, 1].cpu().numpy()
            
    # Final Average
    probs = avg_probs / N_MODELS_S2

    # [FIX] Initialize variables
    best_thresh_found = 0.01
    best_precision_score = 0.0
    
    # PARAMETER: How much do you "care"? 
    # 0.95 = Catch 95% of Exchanges (Standard Safety)
    # 0.98 = Catch 98% of Exchanges (High Safety)
    TARGET_RECALL = 0.88 
    
    print(f"Scanning for Best Threshold (Target Recall >= {TARGET_RECALL:.0%})...")
    print(f"{'Threshold':<10} | {'Recall (Safety)':<15} | {'Precision (Purity)':<18} | {'Flybys Filtered':<15}")
    print("-" * 65)

    # Scan thresholds from 0.01 to 0.95
    for t in np.linspace(0.01, 0.95, 100):
        p_temp = (probs > t).astype(int)
        
        # Confusion Matrix elements
        # True Positive (TP): Exchange correctly predicted as Exchange
        tp = np.sum((p_temp == 1) & (true_labels == 1))
        # False Negative (FN): Exchange missed (classified as Flyby) -> BAD
        fn = np.sum((p_temp == 0) & (true_labels == 1))
        # False Positive (FP): Flyby leaking through (classified as Exchange) -> INEFFICIENCY
        fp = np.sum((p_temp == 1) & (true_labels == 0))
        # True Negative (TN): Flyby correctly filtered
        tn = np.sum((p_temp == 0) & (true_labels == 0))
        
        # Metrics
        recall = tp / (tp + fn + 1e-9)       # Percent of Exchanges caught
        precision = tp / (tp + fp + 1e-9)    # Percent of 'Passed' cases that are actually Exchanges
        
        # CONSTRAINT: We must meet the safety target
        if recall >= TARGET_RECALL:
            # OBJECTIVE: Maximize Precision (i.e., Minimize False Positives/Leaking Flybys)
            if precision > best_precision_score:
                best_precision_score = precision
                best_thresh_found = t
                
                # Print improvement (optional logging)
                # print(f"{t:.3f}      | {recall:.2%}          | {precision:.2%}           | {tn}")

    print("-" * 65)
    print(f"Optimal Threshold Found: {best_thresh_found:.3f}")
    print(f"Safety (Exchange Recall): {np.sum((probs > best_thresh_found) & (true_labels==1)) / np.sum(true_labels==1):.2%}")
    print(f"Purity (Stage 3 Input):   {best_precision_score:.2%}")
    
    # Apply the selected threshold
    base_thresh = best_thresh_found
    strict_thresh = min(1.0, best_thresh_found + 0.20) 
    
    # Apply Conditional Logic
    m1_test = df_test['m1'].values; m2_test = df_test['m2'].values
    mass_ratio = m1_test / (m2_test + 1e-9)

    dynamic_thresholds = np.where(
        (mass_ratio > 0.99) & (mass_ratio < 1.01), 
        strict_thresh, 
        base_thresh
    )
    
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
    
    # [FIX] Plot the line at 'base_thresh' (the one actually used), not the config value
    plt.axvline(base_thresh, color='red', linestyle='--', label=f'Used Threshold ({base_thresh:.2f})')
    
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
    
    