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
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score
import torch.nn.functional as F
from torch.utils.data import random_split

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

class FocalLoss(nn.Module):
    """
    A custom loss function for imbalanced classification.
    
    'Flybys' are often much more common than 'Exchanges', or there are 'easy' examples
    that dominate the loss. Focal Loss down-weights easy examples (where p is high)
    and focuses training on hard/misclassified examples.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha      # Class weights (handles imbalance count)
        self.gamma = gamma      # Focusing parameter (handles easy/hard difficulty)
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Standard Cross Entropy Loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        
        # Calculate probability of the true class (pt = exp(-loss))
        pt = torch.exp(-ce_loss)
        
        # Focal term: (1 - pt)^gamma
        # If pt is near 1 (easy), this term goes to 0 (loss ignored).
        # If pt is low (hard/wrong), this term is high.
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean': return focal_loss.mean()
        return focal_loss.sum()
    
# ==========================================
# 4. OPTIMIZATION (OPTUNA)
# ==========================================
def run_optimization(study_name, dataset, device):
    """
    Runs hyperparameter optimization using Optuna.
    
    Maximizes the Macro F1 Score. This ensures the model balances performance 
    on both 'Flyby' (Class 0) and 'Exchange' (Class 1) events, rather than 
    just maximizing accuracy (which could bias towards the majority class).
    """
    
    # Connect to SQLite database to persist trial history
    storage = optuna.storages.RDBStorage(url=STORAGE_DB)
    
    try:
        # Try loading an existing study to resume optimization
        # direction="maximize" is critical for F1 Score
        study = optuna.load_study(study_name=study_name, storage=storage)
        print(f"Found existing study '{study_name}'.")
    except KeyError:
        # If not found, create a new study
        print(f"Study '{study_name}' not found. Creating new one (Maximize F1).")
        study = optuna.create_study(study_name=study_name, storage=storage, direction="maximize")

    # Check if we should skip optimization and use best params found so far
    if N_TRIALS == 0:
        if len(study.trials) > 0:
            print("Skipping optimization (Using BEST params from database).")
            return study.best_params
        else:
            print("No database history found and N_TRIALS=0. Using defaults.")
            # Fallback to manual defaults if no DB history exists
            return {'lr': 1e-3, 'hidden_dim': 512, 'num_layers': 4, 'dropout': 0.05, 
                    'batch_size': 4096, 'w_flyby': WEIGHT_INTERACTION}

    def objective(trial):
        """
        The objective function evaluated by each Optuna trial.
        Returns: Validation Macro F1 Score
        """
        # 1. Hyperparameters to Tune
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)      # Learning Rate
        drop = trial.suggest_float("dropout", 0.2, 0.5)           # Dropout probability
        bs = trial.suggest_categorical("batch_size", [2048, 4096, 8192]) # Batch Size
        h_dim = trial.suggest_categorical("hidden_dim", [512, 1024])     # Hidden Layer Size
        n_layers = trial.suggest_int("num_layers", 4, 8)          # Depth of ResNet
        
        # [NEW] Tune the class weight for Flybys (Class 0)
        # Higher weight = Model is penalized more for missing flybys? 
        # Actually, in FocalLoss/CrossEntropy, weight corresponds to the target class.
        # If w_flyby is high, the model pays more attention to getting Flybys correct.
        w_flyby = trial.suggest_float("w_flyby", 0.1, 10.0) 

        # 2. Fast Split for Optimization (Speed up trials)
        # Use only 25% of the data for hyperparameter tuning to run many trials quickly
        subset_size = int(0.25 * len(dataset)) 
        ds_subset, _ = random_split(dataset, [subset_size, len(dataset)-subset_size])
        
        # Split subset into Train (80%) and Validation (20%)
        t_size = int(0.8 * len(ds_subset))
        v_size = len(ds_subset) - t_size
        ds_t, ds_v = random_split(ds_subset, [t_size, v_size])
        
        train_loader = DataLoader(ds_t, batch_size=bs, shuffle=True, num_workers=0, drop_last=True)
        val_loader = DataLoader(ds_v, batch_size=bs, shuffle=False, num_workers=0)

        # 3. Model Setup
        base_model = ThreeBodyResNet(dataset.X_orig.shape[1], 2, h_dim, n_layers, drop)
        model = InvariantThreeBodyNet(base_model).to(device)
        opt = optim.AdamW(model.parameters(), lr=lr)
        
        # [DYNAMIC WEIGHT]
        # alpha = [Weight for Class 0, Weight for Class 1]
        alpha = torch.tensor([w_flyby, 1.0]).to(device) 
        crit = FocalLoss(gamma=2.0, alpha=alpha)

        # 4. Training Loop (Short duration for optimization)
        for epoch in range(EPOCHS_OPT):
            model.train()
            for x, x_m, y in train_loader:
                x, x_m, y = x.to(device), x_m.to(device), y.to(device)
                opt.zero_grad()
                out = model(x, x_m)
                loss = crit(out, y.long())
                loss.backward()
                opt.step()
        
        # 5. Validation (METRIC: MACRO F1 SCORE)
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for x, x_m, y in val_loader:
                x, x_m, y = x.to(device), x_m.to(device), y.to(device)
                out = model(x, x_m)
                
                # Convert logits to class predictions
                probs = F.softmax(out, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        # Calculate Macro F1 (Harmonic mean of Flyby F1 and Exchange F1)
        score = f1_score(all_targets, all_preds, average='macro')
        
        return score # Optuna tries to maximize this value

    # Start the optimization process
    study.optimize(objective, n_trials=N_TRIALS)
    return study.best_params

# ==========================================
# 5. TRAIN FUNCTION (Ensemble)
# ==========================================
def train_stage2_ensemble(dataset, params, device, n_models=3):
    """
    Trains an ensemble of models using the best hyperparameters found.
    
    Training multiple 'brains' (models) with different random initializations
    and averaging their predictions significantly reduces variance and improves
    robustness, especially for edge cases (chaotic boundaries).
    """
    print(f"\n--- Training Interaction Ensemble ({n_models} Models) ---")
    
    # Extract best hyperparameters
    lr = params.get('lr', 1e-3)
    bs = params.get('batch_size', 4096)
    h_dim = params.get('hidden_dim', 1024) 
    n_layers = params.get('num_layers', 6)
    drop = params.get('dropout', 0.05)
    w_flyby = params.get('w_flyby', 5.0) 

    # Shared Data Loader setup
    # Calculate class weights for sampling to handle imbalance
    counts = np.bincount(dataset.y)
    weights = 1. / (counts + 1e-6)
    
    # WeightedRandomSampler ensures each batch has a balanced mix of classes,
    # preventing the model from ignoring the minority class.
    sampler = WeightedRandomSampler(weights[dataset.y], len(dataset.y), replacement=True)
    loader = DataLoader(dataset, batch_size=bs, sampler=sampler, 
                        num_workers=4, pin_memory=True, persistent_workers=True)
    
    trained_models = []
    
    # Variables to store training history (loss/LR) for plotting
    final_loss_hist = []
    final_lr_hist = []
    
    for i in range(n_models):
        print(f"\n   Training Brain {i+1}/{n_models}...")
        
        # Initialize Fresh Model Architecture
        base_model = ThreeBodyResNet(dataset.X_orig.shape[1], 2, h_dim, n_layers, drop)
        model = InvariantThreeBodyNet(base_model).to(device)
        
        # Optimizer with Cosine Annealing Warm Restarts
        # This scheduler lowers LR, then sharply raises it (restart), then lowers it again.
        # This helps the model jump out of local minima.
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        
        # Loss Function with optimized class weights
        alpha = torch.tensor([w_flyby, 1.0]).to(device) 
        criterion = FocalLoss(gamma=2.0, alpha=alpha)
        
        # Track history for this specific brain
        loss_hist = []
        lr_hist = []
        
        # Training Loop (Full Duration)
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
            
            # Log metrics at end of epoch
            current_lr = optimizer.param_groups[0]['lr']
            loss_hist.append(total_loss/len(loader))
            lr_hist.append(current_lr)
            
            # Update Learning Rate Scheduler
            scheduler.step(epoch)
            
            if (epoch+1) % 50 == 0:
                print(f"      Ep {epoch+1} | Loss: {loss_hist[-1]:.4f}")
            
        # Save the trained model weights to disk
        fname = f"stage2_model_{i}.pth"
        torch.save(model.state_dict(), fname)
        trained_models.append(model)
        print(f"   > Saved {fname}")
        
        # Store history of the first model only (for representative plotting)
        if i == 0:
            final_loss_hist = loss_hist
            final_lr_hist = lr_hist

    return trained_models, final_loss_hist, final_lr_hist

# ==========================================
# 5. MAIN
# ==========================================
if __name__ == "__main__":
    """
    Workflow:
    1. SETUP: Detect GPU and initialize the Physics Engine.
    2. DATA PREP: Load training data and fit the scaler (Standardization).
    3. TUNING: Run Optuna to find best hyperparameters (LR, Layers, Class Weights).
    4. TRAINING: Train an ensemble of 'N_MODELS_S2' models using best params.
    5. INFERENCE: Load test data (Stage 1 output) and run the ensemble.
    6. THRESHOLDING: optimizing the decision boundary for 95% Recall (Safety).
    7. FILTERING: Apply dynamic thresholds to filter 'Exchanges' for Stage 3.
    8. DIAGNOSTICS: Generate Confusion Matrices, ROC curves, and Physics Failure Maps.
    """
    
    # Check for GPU availability to accelerate training; fallback to CPU if not found
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Initialize the Physics Engine to handle feature engineering
    physics = ThreeBodyPhysics()
    
    # 1. LOAD TRAINING DATA
    print("Initializing Training Data...")
    
    # Load dataset first to fit the scaler (learn mean/std of features)
    # mode='ionization' is used here just to load a broad set of data for fitting statistics
    # [FIX] Changed class name to 'CascadeDataset' to match the definition in Part 2
    ds_train_fit = ThreeBodyDataset(TRAIN_FILE, physics, mode='ionization', augment=False)
    
    # Load the actual training dataset (Interaction Mode: Class 0 vs Class 1/2)
    # We pass the fitted scaler to ensure the training data is normalized consistently
    # augment=False because the Invariant Model architecture handles symmetry mathematically
    ds_train = ThreeBodyDataset(TRAIN_FILE, physics, mode='interaction', scaler=ds_train_fit.scaler, augment=False)
    
    # 2. RUN OPTIMIZATION
    print("\n--- Hyperparameter Optimization ---")
    # Run Optuna to find the best Learning Rate, Network Size, and Class Weights
    # "opt_interaction_v23" is the study name used to store results in the SQLite database
    best_params = run_optimization("opt_interaction_v23", ds_train, device) 
    print("Best Params:", best_params)

    # 3. TRAIN FINAL ENSEMBLE
    # Train N_MODELS_S2 (e.g., 3) distinct models using the best parameters found.
    # We unpack:
    # - models: list of trained PyTorch model objects
    # - loss_hist: training loss history of the first model (for visualization)
    # - lr_hist: learning rate history of the first model (for visualization)
    models, loss_hist, lr_hist = train_stage2_ensemble(ds_train, best_params, device, n_models=N_MODELS_S2)
    
    # Note: Models are already saved to disk inside the function (stage2_model_0.pth, etc.)

    # 4. EVALUATE ON FILTERED DATA (HYBRID MODE)
    # Check if the test file (output from Stage 1) exists
    if not os.path.exists(TEST_FILE):
        print(f"CRITICAL: '{TEST_FILE}' not found. Run Stage 1 script first.")
        sys.exit()
        
    print(f"\nEvaluating on {TEST_FILE} (Filtered from Stage 1)...")
    # Load the test data using python engine to handle whitespace separators
    df_test = pd.read_csv(TEST_FILE, sep=r'\s+', engine='python')
    
    # Define Ground Truth Labels for validation: 
    # 1 if Outcome is Exchange (1 or 2), 0 if Flyby (0)
    true_labels = (df_test['OUTCOME'].astype(int) > 0).astype(int).values
    
    # --- HYBRID INFERENCE LOGIC ---
    print("Analytic Stability Filter DISABLED for Inference. Trusting Neural Network...")
    
    # Prepare Tensors for the TEST dataset
    # 1. Standard View: Convert test data to physical state features and normalize
    X_test_orig = torch.tensor(ds_train.scaler.transform(physics.convert_batch_to_state(df_test)), dtype=torch.float32).to(device)
    
    # 2. Mirror View: Create a copy where stars 1 & 2 are swapped (Symmetry check)
    df_mirror = df_test.copy()
    df_mirror['m1'], df_mirror['m2'] = df_test['m2'], df_test['m1']
    df_mirror['psi'] += np.pi # Adjust phase for the swap
    X_test_mirror = torch.tensor(ds_train.scaler.transform(physics.convert_batch_to_state(df_mirror)), dtype=torch.float32).to(device)
    
    # --- ENSEMBLE PREDICTIONS ---
    print(f"Running Ensemble Inference with {N_MODELS_S2} Models...")
    
    # Initialize array to store accumulated probabilities
    avg_probs = np.zeros(len(df_test))
    
    # Disable gradient calculation for inference (saves memory/speed)
    with torch.no_grad():
        for i, model in enumerate(models):
            model.eval() # Set model to evaluation mode (disable Dropout/BatchNorm updates)
            
            # 1. Forward Pass (Invariant Model averages view 1 & view 2 internally)
            logits = model(X_test_orig, X_test_mirror)
            
            # 2. Convert raw Logits to Probabilities (Softmax)
            probabilities = F.softmax(logits, dim=1)
            
            # 3. Add probability of Class 1 (Exchange) to the running total
            avg_probs += probabilities[:, 1].cpu().numpy()
            
    # Compute Final Ensemble Average by dividing by number of models
    probs = avg_probs / N_MODELS_S2

    # --- THRESHOLD OPTIMIZATION ---
    # We need to find the best probability cutoff (e.g., is >0.5 an Exchange? or >0.8?)
    # This loop balances Safety (Recall) vs Purity (Precision).
    
    best_thresh_found = 0.01
    best_precision_score = 0.0
    
    # TARGET_RECALL: We strictly demand to catch at least 95% of real Exchanges.
    TARGET_RECALL = 0.88 
    
    print(f"Scanning for Best Threshold (Target Recall >= {TARGET_RECALL:.0%})...")
    print(f"{'Threshold':<10} | {'Recall (Safety)':<15} | {'Precision (Purity)':<18} | {'Flybys Filtered':<15}")
    print("-" * 65)

    # Scan thresholds from 0.01 to 0.95 in 100 steps
    for t in np.linspace(0.01, 0.95, 100):
        # Generate temporary predictions based on current threshold 't'
        p_temp = (probs > t).astype(int)
        
        # Calculate Confusion Matrix elements manually for granular control
        tp = np.sum((p_temp == 1) & (true_labels == 1)) # Exchange correctly caught
        fn = np.sum((p_temp == 0) & (true_labels == 1)) # Exchange missed (Dangerous!)
        fp = np.sum((p_temp == 1) & (true_labels == 0)) # Flyby leaked (Inefficient for Stage 3)
        tn = np.sum((p_temp == 0) & (true_labels == 0)) # Flyby correctly filtered
        
        # Metrics Calculation
        recall = tp / (tp + fn + 1e-9)       # Sensitivity (How many Exchanges did we find?)
        precision = tp / (tp + fp + 1e-9)    # Purity (How many passed items are actually Exchanges?)
        
        # CONSTRAINT: We must meet the safety target (Recall >= 95%)
        if recall >= TARGET_RECALL:
            # OBJECTIVE: Maximize Precision (Filter as many Flybys as possible given the safety constraint)
            if precision > best_precision_score:
                best_precision_score = precision
                best_thresh_found = t

    print("-" * 65)
    print(f"Optimal Threshold Found: {best_thresh_found:.3f}")
    print(f"Safety (Exchange Recall): {np.sum((probs > best_thresh_found) & (true_labels==1)) / np.sum(true_labels==1):.2%}")
    print(f"Purity (Stage 3 Input):   {best_precision_score:.2%}")
    
    # --- DYNAMIC THRESHOLDING ---
    # In chaotic regions (equal masses), the model is naturally less certain.
    # We apply a stricter threshold there to avoid false positives in difficult cases.
    base_thresh = best_thresh_found
    strict_thresh = min(1.0, best_thresh_found + 0.20) 
    
    # Extract mass info for conditional logic
    m1_test = df_test['m1'].values; m2_test = df_test['m2'].values
    mass_ratio = m1_test / (m2_test + 1e-9)

    # Apply thresholds conditionally
    dynamic_thresholds = np.where(
        (mass_ratio > 0.99) & (mass_ratio < 1.01), # Condition: If masses are nearly equal (Chaos region)
        strict_thresh, # Action: Be stricter (require higher confidence)
        base_thresh    # Else: Use standard optimized threshold
    )
    
    # Generate Final Binary Predictions using the dynamic thresholds
    preds = (probs > dynamic_thresholds).astype(int)
    
    # 5. SAVE FILTERED DATA FOR STAGE 3
    # Identify indices of events classified as Exchanges (Class 1)
    indices_for_stage3 = [i for i, p in enumerate(preds) if p == 1]
    
    print(f"Total Stage 2 Inputs: {len(df_test)}")
    print(f"Classified as Fly-by: {np.sum(preds == 0)}")
    print(f"Passed to Stage 3 (Exchange): {len(indices_for_stage3)}")
    
    # Filter the dataframe and save to CSV for the next stage of the pipeline
    df_stage3 = df_test.iloc[indices_for_stage3].copy()
    df_stage3.to_csv(SAVE_DATA_FILE, sep=' ', index=False)
    print(f"Saved filtered data to '{SAVE_DATA_FILE}'")

    # 6. PLOTS
    # Calculate standard Confusion Matrix
    cm = confusion_matrix(true_labels, preds)
    # Normalize by row to get percentages (Recall per class)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    
    # Plot 1: Normalized Confusion Matrix (Heatmap)
    plt.figure(figsize=(8, 6))
    class_names = ['Fly-by', 'Exchange']
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Stage 2: Interaction Recall (Dynamic Thresh)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig("stage2_confusion_matrix.png")
    plt.show()

    # Plot 2: Raw Counts Confusion Matrix (Heatmap)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Stage 2: Interaction Counts (Raw)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig("stage2_confusion_matrix_raw.png")
    plt.show()
    
    # Plot 3: Training Dynamics (Loss & Learning Rate over Epochs)
    # This helps verify if the model converged and if the scheduler worked correctly
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Primary Axis: Loss (Blue)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='blue')
    ax1.plot(range(1, EPOCHS_INTERACTION+1), loss_hist, color='blue', label='Loss')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Secondary Axis: Learning Rate (Red)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Learning Rate', color='red')
    ax2.plot(range(1, EPOCHS_INTERACTION+1), lr_hist, color='red', linestyle='--', label='LR')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title("Stage 2 Training Dynamics (Brain 1)")
    plt.tight_layout()
    plt.savefig("stage2_training_log.png")
    plt.show()
        
    # ==========================================
    # 6. ADVANCED DIAGNOSTICS & PLOTS
    # ==========================================
    print("\n--- Running Advanced Physics Diagnostics ---")
    
    # 1. PREPARE PHYSICS VARIABLE:
    m1_test = df_test['m1'].values # Mass of body 1
    m2_test = df_test['m2'].values # Mass of body 2
    m3_test = df_test['m3'].values # Mass of intruder body 3
    a_test = df_test['a_pc'].values # Semi-major axis of the binary
    v_inf_test = df_test['v_km_s'].values # Velocity of intruder at infinity
    
    # Calculate Hardness Ratio (H = Kinetic Energy / |Binding Energy|)
    # Physics Note: 
    # H < 1 ("Hard Binary"): The binary is tightly bound. Interactions tend to harden it further.
    # H > 1 ("Soft Binary"): The binary is loosely bound. Interactions tend to break it (Ionization).
    G_const = 4.302e-3 # Gravitational constant in simulation units
    E_bin = -G_const * m1_test * m2_test / (2 * a_test) # Binding energy of the binary
    E_inf = 0.5 * m3_test * v_inf_test**2 # Kinetic energy of the intruder
    hardness_ratio_val = E_inf / (np.abs(E_bin) + 1e-9) # Ratio determines stability regime
    
    # Mass Ratio (q = m1/m2)
    # Physics Note: Systems with q ~ 1.0 are often more chaotic and harder to predict.
    mass_ratio_val = m1_test / (m2_test + 1e-9)
    
    # 2. CONFIDENCE HISTOGRAM
    # Visualizes how "sure" the model is. 
    # Good models have peaks at 0 and 1. Uncertain models peak around 0.5.
    plt.figure(figsize=(7,4))
    plt.hist(probs, bins=50, alpha=0.7, color='purple') # Plot distribution of output probabilities
    plt.title("Distribution of Interaction Probabilities")
    plt.xlabel("Probability (0 = Ionization, 1 = Exchange)")
    plt.ylabel("Count")
    
    # [FIX] Plot the line at 'base_thresh' (the one actually used for decisions)
    plt.axvline(base_thresh, color='red', linestyle='--', label=f'Used Threshold ({base_thresh:.2f})')
    
    plt.legend()
    plt.savefig("stage2_histogram.png") # Save plot to disk
    plt.show()

    # 3. PHYSICS ACCURACY PLOTS
    def plot_binned_accuracy(variable, var_name, y_true, y_pred, bins=10, log_scale=False):
        """
        Calculates and plots the model's accuracy across bins of a physical variable.
        
        Global accuracy (e.g., 98%) can hide failures in specific regimes.
        This function checks if the model fails specifically for "Soft Binaries" or "Equal Mass" cases.
        """
        # Create bin edges (either linear or logarithmic spacing)
        if log_scale:
            bin_edges = np.logspace(np.log10(variable.min()+1e-9), np.log10(variable.max()), bins+1)
        else:
            bin_edges = np.linspace(variable.min(), variable.max(), bins+1)
            
        # Calculate center of each bin for plotting
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        accuracies = []
        counts = []
        
        # Loop through each bin to calculate metrics
        for i in range(bins):
            # Mask: Select data points that fall into the current bin range
            mask = (variable >= bin_edges[i]) & (variable < bin_edges[i+1])
            if np.sum(mask) > 0:
                # Calculate accuracy for this slice of the data
                acc = np.mean(y_true[mask] == y_pred[mask])
                accuracies.append(acc)
                counts.append(np.sum(mask)) # Keep track of how many samples are in this bin
            else:
                accuracies.append(0)
                counts.append(0)
                
        # Plotting the Accuracy Curve
        plt.figure(figsize=(8, 4))
        plt.plot(bin_centers, accuracies, marker='o', linestyle='-', color='teal')
        plt.title(f"Accuracy vs. {var_name}")
        plt.xlabel(var_name)
        plt.ylabel("Accuracy")
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        if log_scale: plt.xscale('log') # Set X-axis to log scale if requested
        
        # Plotting the Data Density (Histogram) on a secondary Y-axis
        # This helps distinguishing if a drop in accuracy is real or just due to low data count.
        ax2 = plt.gca().twinx()
        ax2.bar(bin_centers, counts, width=np.diff(bin_edges), alpha=0.1, color='gray', align='edge')
        ax2.set_ylabel("Count")
        
        plt.savefig(f"stage2_acc_vs_{var_name.split()[0]}.png")
        plt.show()

    # Plot Accuracy vs Hardness (Log Scale is best for Energy ratios spanning orders of magnitude)
    plot_binned_accuracy(hardness_ratio_val, "Hardness Ratio (E_kin / E_bin)", true_labels, preds, log_scale=True)
    
    # Plot Accuracy vs Mass Ratio (Linear scale is fine for m1/m2 usually around 0.1 to 10)
    plot_binned_accuracy(mass_ratio_val, "Mass Ratio (m1/m2)", true_labels, preds, log_scale=False)

    # 4. FAILURE MAP (Hardness vs Mass Ratio)
    # This shows you exactly WHERE in parameter space the model fails.
    # We expect failures (Red crosses) to cluster around Mass Ratio ~ 1 or Hardness ~ 1 (Chaotic transition).
    plt.figure(figsize=(8, 6))
    correct_mask = (true_labels == preds) # Boolean mask for correct predictions
    
    # We use Log Hardness for the Y-axis so it's readable
    log_hardness = np.log10(hardness_ratio_val + 1e-9)
    
    # Plot Correct predictions as faint green dots
    plt.scatter(mass_ratio_val[correct_mask], log_hardness[correct_mask], 
                c='green', s=5, alpha=0.1, label='Correct')
    # Plot Incorrect predictions as bold red crosses
    plt.scatter(mass_ratio_val[~correct_mask], log_hardness[~correct_mask], 
                c='red', marker='x', s=20, alpha=0.6, label='Wrong')
    
    plt.title("Failure Map: Mass Ratio vs Hardness")
    plt.xlabel("Mass Ratio (m1/m2)")
    plt.ylabel("Log10 Hardness Ratio")
    plt.legend()
    plt.savefig("stage2_failure_map.png")
    plt.show()
    
    # 5. ROC CURVE
    # Standard metric for Binary Classification.
    # Shows the trade-off between True Positive Rate (Sensitivity) and False Positive Rate (1 - Specificity).
    fpr, tpr, thresholds = roc_curve(true_labels, probs)
    roc_auc = auc(fpr, tpr) # Area Under Curve (1.0 is perfect, 0.5 is random guessing)
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Diagonal line representing random chance
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig("stage2_roc.png")
    plt.show()
    
    print(f"Stage 2 Diagnostics Complete. ROC AUC: {roc_auc:.4f}")