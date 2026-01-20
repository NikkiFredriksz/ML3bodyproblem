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
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score
import torch.nn.functional as F

# ==========================================
# CONFIGURATION
# ==========================================
# INPUTS
STORAGE_DB = "sqlite:///three_body_stage_1_v4.db"  # Database connection string for Optuna studies
TRAIN_FILE = "train3body.dat"                      # Filename for the training dataset
TEST_FILE = "test3body.dat"                        # Filename for the testing dataset

# OUTPUTS
SAVE_MODEL_FILE = "stage1_ion_v23.pth"             # Filename to save the trained PyTorch model weights
SAVE_DATA_FILE = "data_for_stage2.csv"             # Filename to save difficult cases for the next stage

# SETTINGS (Aggressive Ionization Filtering)
N_TRIALS = 100                                     # Number of hyperparameter optimization trials to run
EPOCHS_OPT = 50                                    # Number of training epochs during optimization
EPOCHS_IONIZATION = 30                             # Number of epochs for the final training run
WEIGHT_IONIZATION = 1.0                            # Loss weight: 1.0 treats Ionization/Bound classes equally
THRESH_IONIZATION = 0.4                            # Probability threshold: >0.4 triggers an Ionization prediction

# ==========================================
# 1. PHYSICS ENGINE
# ==========================================
class ThreeBodyPhysics:
    """
    Encapsulates all domain knowledge, coordinate transformations, and physical laws.
    
    Raw data (mass, position) is hard for a neural network to interpret directly.
    This class converts raw inputs into physically meaningful features (Energy, 
    Angular Momentum, Phase Angles) which makes learning significantly faster and 
    more accurate. It also handles the 'Physics Veto' to rule out impossible cases.
    """
    def __init__(self): 
        self.G = 4.302e-3  # Gravitational Constant in units (pc * km^2 / s^2 / M_sun)

    def _calculate_core_physics(self, df):
        """
        Internal helper to calculate shared physics variables once.
        
        Energy calculations are used in multiple places (filtering and feature generation).
        Calculating them once here avoids code duplication and potential math errors.
        """
        # Extract raw columns from the dataframe as numpy arrays
        m1 = df['m1'].values; m2 = df['m2'].values; m3 = df['m3'].values
        a = df['a_pc'].values; v_inf = df['v_km_s'].values
        
        # Energy Calculation
        # Potential Energy of the inner binary (negative because it's bound)
        E_bin = -self.G * m1 * m2 / (2 * a)
        # Kinetic Energy of the incoming third star at infinity
        E_inf = 0.5 * m3 * v_inf**2
        # Total Energy of the 3-body system (Conserved quantity)
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
        """
        Converts raw simulation data into a feature vector for the Neural Network.
        
        This transforms 2D orbital elements (a, e, i) into 3D Cartesian vectors 
        and calculates advanced metrics like Angular Momentum vectors and Phase angles.
        It provides the 'State' of the system at the exact moment of interaction.
        """
        # Use the internal helper to get basic vars
        m1, m2, m3, a, v_inf, E_bin, E_inf, E_tot = self._calculate_core_physics(df)
        
        # Calculate hardness ratio again for the feature list
        hardness_ratio = E_inf / (np.abs(E_bin) + 1e-9)
        
        # 1. Unpack Variables
        e = df['e'].values; b = df['b_pc'].values  # Eccentricity and Impact Parameter
        
        # Keep angles raw for rotation logic
        # np.where handles normalizing angles to range [-pi, pi] if the data is messy
        phi = np.where(np.abs(df['phi'].values)>2*np.pi, np.radians(df['phi'].values), df['phi'].values)
        theta = np.where(np.abs(df['theta'].values)>2*np.pi, np.radians(df['theta'].values), df['theta'].values)
        psi = np.where(np.abs(df['psi'].values)>2*np.pi, np.radians(df['psi'].values), df['psi'].values)
        f = df['f'].values; t_coal = df['t_coal_yr'].values # True anomaly and Coalescence time
        
        # --- FEATURE ENGINEERING ---
        M_bin = m1 + m2             # Mass of the binary
        M_tot = m1 + m2 + m3        # Total mass of the system
        
        # --- PHASE FEATURES ---
        # Predicting the state of the binary when the third star actually arrives
        r_peri_encounter = b  # Approximation: Closest approach is roughly the impact parameter
        # Vis-viva equation to find velocity at closest approach
        v_peri_encounter = np.sqrt(v_inf**2 + 2*self.G*M_bin/(r_peri_encounter+1e-9))
        v_avg = np.sqrt(v_inf * v_peri_encounter) # Geometric mean velocity
        t_approach = (50.0 * a) / (v_avg + 1e-9)  # Time it takes for star 3 to arrive (starting from 50*a)

        mean_motion = np.sqrt(self.G * M_bin / (a**3 + 1e-9)) # Angular speed of the binary
        # Projecting the binary phase (f) forward by time (t_approach)
        M_encounter = f + mean_motion * t_approach
        
        # Store Sin/Cos of the encounter phase (Neural nets handle trig better than raw angles)
        feat_phase_sin = np.sin(M_encounter)
        feat_phase_cos = np.cos(M_encounter)
        
        # --- COORDINATE TRANSFORMS ---
        # 1. Calculate position/velocity in the 2D orbital plane
        r_mag = (a * (1 - e**2)) / (1 + e * np.cos(f)) # Distance between m1 and m2
        term_h = self.G * M_bin * a * (1 - e**2)       # Specific angular momentum term
        h_spec = np.sqrt(np.maximum(0.0, term_h))      # Specific angular momentum
        inv_h = np.zeros_like(h_spec); mask_h = h_spec > 0
        inv_h[mask_h] = 1.0 / h_spec[mask_h]           # Inverse h (safe division)
        
        vr = (self.G * M_bin * e * np.sin(f)) * inv_h  # Radial velocity component
        vt = h_spec / (r_mag + 1e-9)                   # Tangential velocity component
        
        c_f, s_f = np.cos(f), np.sin(f)
        # Position vector in 2D plane
        r_rel_plane = np.stack([r_mag * c_f, r_mag * s_f, np.zeros_like(f)], axis=1)
        # Velocity vector in 2D plane
        v_rel_plane = np.stack([vr * c_f - vt * s_f, vr * s_f + vt * c_f, np.zeros_like(f)], axis=1)
        
        # 2. Rotate 2D Plane into 3D Space using Euler Angles (phi, theta, psi)
        z = np.zeros_like(phi); o = np.ones_like(phi) # Zeros and Ones helpers
        # Rotation Matrix Z (Phi)
        c, s = np.cos(phi), np.sin(phi); Rz_phi = np.stack([np.stack([c,-s,z],1), np.stack([s,c,z],1), np.stack([z,z,o],1)],1)
        # Rotation Matrix X (Theta)
        c, s = np.cos(theta), np.sin(theta); Rx_theta = np.stack([np.stack([o,z,z],1), np.stack([z,c,-s],1), np.stack([z,s,c],1)],1)
        # Rotation Matrix Z (Psi)
        c, s = np.cos(psi), np.sin(psi); Rz_psi = np.stack([np.stack([c,-s,z],1), np.stack([s,c,z],1), np.stack([z,z,o],1)],1)
        # Combined Rotation Matrix
        R = Rz_phi @ Rx_theta @ Rz_psi
        
        # Apply rotation to get final 3D vectors for the binary
        r_rel = (R @ r_rel_plane[:,:,None]).squeeze(-1)
        v_rel = (R @ v_rel_plane[:,:,None]).squeeze(-1)
        
        # Define 3D vectors for the incoming third star (approaching along X-axis approximation)
        r3 = np.stack([50*a, b, np.zeros_like(a)], axis=1) # Starts at 50*a distance, offset by b
        v3 = np.stack([-v_inf, np.zeros_like(v_inf), np.zeros_like(v_inf)], axis=1) # Moves negative x direction

        # --- ANGULAR MOMENTUM CALCS ---
        # Cross product r x v gives Angular Momentum (L) vectors
        L_bin_vec = np.cross(r_rel, v_rel) # L of the binary
        L_outer_vec = np.cross(r3, v3)     # L of the third star relative to binary
        
        # NEW: Total Angular Momentum Magnitude
        L_tot_vec = L_bin_vec + L_outer_vec
        L_tot_mag = np.linalg.norm(L_tot_vec, axis=1)

        # Inclination (Angle between the two angular momentum vectors)
        dot_L = np.sum(L_bin_vec * L_outer_vec, axis=1)
        norm_L = np.linalg.norm(L_bin_vec, axis=1) * np.linalg.norm(L_outer_vec, axis=1)
        cos_inclination = dot_L / (norm_L + 1e-9)

        # Momentum Ratios (Comparison of strengths)
        mu_bin = m1 * m2 / (M_bin + 1e-9)
        L_bin_mag = mu_bin * np.sqrt(self.G * M_bin * a * (1 - e**2) + 1e-9)
        mu_out = m3 * M_bin / (M_tot + 1e-9)
        L_inf_mag = mu_out * b * (v_inf + 1e-9)
        L_ratio = L_inf_mag / (L_bin_mag + 1e-9) # Ratio of Outer L to Inner L

        # Log-Modulus helper: allows taking log of negative numbers by preserving sign
        def lm(x): return np.sign(x)*np.log10(1+np.abs(x))
        
        # Compactness: A metric of how "small" the binary is compared to interaction speed
        r_peri = a * (1 - e)
        compactness = M_tot / (r_peri * (v_inf**2 + 1e-6) + 1e-9)
        
        # Calculate distances between individual stars
        rm2 = (m2/M_bin)[:,None]; rm1 = (m1/M_bin)[:,None]
        r1 = -rm2*r_rel; r2 = rm1*r_rel
        d13 = np.linalg.norm(r1 - r3, axis=1) # Distance star 1 to star 3
        d23 = np.linalg.norm(r2 - r3, axis=1) # Distance star 2 to star 3
        
        # Stack all calculated features into a list
        feat = [
            np.log10(m1)[:,None], np.log10(m2)[:,None], np.log10(m3)[:,None], # Log Masses
            np.log10(a)[:,None],                                              # Log Semi-major axis
            np.log10(np.maximum(1e-9, t_coal))[:,None],                       # Log Coalescence time
            (m1/m2)[:,None], (m2/m3)[:,None], (m3/m1)[:,None],                # Mass Ratios
            
            # THE IONIZATION PREDICTORS
            lm(E_tot)[:,None],                   # Total Energy (Sign-preserved Log)
            np.log10(hardness_ratio)[:,None],    # Hardness Ratio
            
            np.log10(L_ratio)[:,None],           # Angular Momentum Ratio
            np.log10(L_tot_mag + 1e-9)[:,None],  # Total Angular Momentum (NEW)
            
            np.log10(r_peri + 1e-9)[:,None],     # Log Pericenter distance
            np.log10((m3/M_bin)*(a/(b+1e-9))**3+1e-9)[:,None], # Tidal Strength Approximation
            np.sin(f)[:,None], np.cos(f)[:,None],              # Binary Phase
            feat_phase_sin[:,None], feat_phase_cos[:,None],    # Projected Interaction Phase
            lm(d13-d23)[:, None],                              # Asymmetry in distances to intruder
            cos_inclination[:, None],                          # Cosine of Inclination
            np.log10(compactness + 1e-9)[:,None]               # System compactness
        ]
        # Stack horizontally to create a (Batch_Size, N_Features) matrix
        return np.hstack(feat).astype(np.float32)

# ==========================================
# 2. DATASET
# ==========================================
class ThreeBodyDataset(Dataset):
    """
    Custom PyTorch Dataset that loads, filters, balances, and transforms simulation data.
    
    1. FILTERING: It implements the 'Hard Negative Mining' strategy by removing 
       physically impossible cases (E < 0) AND easy fly-bys (r_min > 5a).
       This forces the AI to train ONLY on the difficult 'Chaotic Core'.
    2. BALANCING: Ionization events are rare (~1-5%). This class duplicates them
       (Oversampling) so the AI sees a 50/50 split during training.
    3. SYMMETRY: It generates a 'Mirror State' (swapping star 1 & 2) for every sample.
       This allows the Siamese Network to learn that physics is symmetric.
    """
    def __init__(self, filepath, physics_engine, mode='ionization', scaler=None, augment=True):
        if not os.path.exists(filepath): sys.exit(f"Error: {filepath} not found.")
        
        # 1. Load Data from text file into Pandas DataFrame
        data = pd.read_csv(filepath, sep=r'\s+', engine='python')
        
        # 2. PHYSICS PRE-FILTER (Standard)
        # Use the Physics Engine to identify physically impossible scenarios (Energy < 0)
        # We discard these immediately because the answer is trivially "Bound" (0).
        is_possible, _ = physics_engine.get_physics_flags(data)
        data = data[is_possible].copy()
        
        # ---------------------------------------------------------
        # 3. CHAOTIC CORE FILTERING
        # ---------------------------------------------------------
        # Calculate the True Closest Approach (r_min) accounting for Gravitational Focusing.
        # WHY: Simple impact parameter (b) is misleading for slow stars that get pulled in.
        # We use Hyperbolic Orbit Mechanics to find the actual periastron distance.
        
        G = 4.302e-3  # Gravitational constant
        M_tot = data['m1'] + data['m2'] + data['m3'] # Total mass
        v_inf = data['v_km_s'] # Velocity at infinity
        b = data['b_pc']       # Impact parameter at infinity
        
        # Calculate Hyperbolic Semi-Major Axis (a_hyp)
        # Represents the scale of the hyperbolic encounter orbit.
        # Note: Adding 1e-9 to v_inf to avoid division by zero errors
        a_hyp = (G * M_tot) / (v_inf**2 + 1e-9)
        
        # Calculate Eccentricity of the encounter orbit
        # e > 1 for hyperbolic orbits.
        e_hyp = np.sqrt(1 + (b / a_hyp)**2)
        
        # True Closest Approach Distance (r_min)
        # This is how close the third star actually gets to the binary center of mass.
        r_min = a_hyp * (e_hyp - 1)
        
        # Logic: If the star actually gets within 5 binary radii, it's dangerous (Close Encounter).
        # If r_min > 5a, tidal forces are too weak to cause ionization.
        is_flyby = r_min > (5.0 * data['a_pc'])
        
        # ---------------------------------------------------------
        # Filter Mask:
        # Keep the sample IF:
        # 1. It is NOT a safe fly-by (it's a dangerous close encounter)
        #    OR
        # 2. It resulted in Ionization (OUTCOME == 3)
        #    (We always keep ionizations, even if they look like fly-bys, to capture rare edge cases)
        mask_chaotic = (~is_flyby) | (data['OUTCOME'] == 3)
        
        n_dropped = len(data) - np.sum(mask_chaotic)
        data = data[mask_chaotic].copy()
        
        print(f"[Dataset] Chaotic Core: Dropped {n_dropped} safe fly-bys (Calculated via r_min).")
        
        # ---------------------------------------------------------
        
        # 4. Get Labels
        # Convert Outcome column (1,2,3) into Binary Target (0=Bound, 1=Ionization)
        raw_outcomes = data['OUTCOME'].astype(int).values
        y_raw = (raw_outcomes == 3).astype(int) # 1 if Ionization, 0 otherwise
        
        # 5. Oversampling (Balancing the Class Distribution)
        # Since Ionization is rare, we duplicate the '1' samples until they match the count of '0's.
        dfs = [data]
        ys = [y_raw]
        
        if augment and mode == 'ionization':
            mask_ion = (y_raw == 1)
            count_ion = np.sum(mask_ion)
            count_bound = len(y_raw) - count_ion
            
            if count_ion > 0:
                # Calculate how many times we need to duplicate the Ionization samples
                multiplier = int(count_bound / count_ion) - 1
                if multiplier > 0:
                    df_ion = data[mask_ion].copy()
                    y_ion = y_raw[mask_ion].copy()
                    for _ in range(multiplier):
                        dfs.append(df_ion)
                        ys.append(y_ion)
                    print(f"[Oversampling] Balanced with {multiplier}x duplication.")

        # Combine original data + duplicates into final dataset
        self.df_final = pd.concat(dfs, ignore_index=True)
        self.y = np.concatenate(ys)
        print(f"[Dataset] Final Size: {len(self.y)} (Focusing on Hard Cases)")

        # 6. Generate States (Feature Extraction)
        # Convert the DataFrame rows into PyTorch-ready feature vectors
        self.X_orig = physics_engine.convert_batch_to_state(self.df_final)
        
        # Generate Mirror State (Symmetry Augmentation)
        # Create a copy where Star 1 and Star 2 are swapped
        df_mirror = self.df_final.copy()
        df_mirror['m1'], df_mirror['m2'] = self.df_final['m2'], self.df_final['m1']
        df_mirror['psi'] += np.pi # Rotate system by 180 degrees to maintain geometry
        self.X_mirror = physics_engine.convert_batch_to_state(df_mirror)
        
        # Scaling (Normalization)
        # If a scaler is provided (e.g., from training set), use it.
        # If not, fit a new StandardScaler to normalize features (mean=0, std=1).
        if scaler:
            self.X_orig = scaler.transform(self.X_orig)
            self.X_mirror = scaler.transform(self.X_mirror)
            self.scaler = scaler
        else:
            self.scaler = StandardScaler()
            # Fit on BOTH original and mirror states to ensure symmetry in scaling
            combined = np.concatenate([self.X_orig, self.X_mirror], axis=0)
            self.scaler.fit(combined)
            self.X_orig = self.scaler.transform(self.X_orig)
            self.X_mirror = self.scaler.transform(self.X_mirror)

    def __len__(self): 
        """Returns total number of samples."""
        return len(self.y)

    def __getitem__(self, idx): 
        """
        Returns a single sample tuple:
        1. X_orig: The standard feature vector
        2. X_mirror: The symmetric twin (swapped m1/m2)
        3. y: The label (0 or 1)
        """
        return (torch.tensor(self.X_orig[idx]), 
                torch.tensor(self.X_mirror[idx]), 
                torch.tensor(self.y[idx], dtype=torch.long))

# ==========================================
# 3. MODEL
# ==========================================

class ResidualBlock(nn.Module):
    """
    A standard building block for Deep Residual Networks (ResNets).
    
    As neural networks get deeper (more layers), they become harder to train because
    gradients 'vanish' (shrink to zero) during backpropagation. 
    This block adds a 'skip connection' (x + block(x)) which allows information 
    and gradients to flow through the network unimpeded. This is essential for 
    learning complex, non-linear physics relationships without getting stuck.
    """
    def __init__(self, hidden_dim, dropout_rate):
        super(ResidualBlock, self).__init__()
        # Define the stack of layers:
        self.block = nn.Sequential(
            # Linear transformation (Dense layer)
            nn.Linear(hidden_dim, hidden_dim), 
            # Normalizes inputs to mean=0, std=1. Stabilizes training and allows higher learning rates.
            nn.BatchNorm1d(hidden_dim), 
            # GELU (Gaussian Error Linear Unit): A modern activation function, smoother than ReLU.
            nn.GELU(), 
            # Randomly zeros out neurons to prevent overfitting (the AI memorizing specific examples).
            nn.Dropout(dropout_rate),
            
            # Repeat the pattern for depth
            nn.Linear(hidden_dim, hidden_dim), 
            nn.BatchNorm1d(hidden_dim), 
            nn.GELU(), 
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, x): 
        # The "Residual Connection": Add the original input 'x' to the processed output.
        # This allows the network to learn identity functions easily.
        return x + self.block(x)

class InvariantThreeBodyNet(nn.Module):
    """
    A 'Siamese' wrapper that enforces Physical Symmetry (Permutation Invariance).
    
    In a hierarchical triple system, the inner binary consists of Star 1 and Star 2.
    Physics doesn't care which one you label "1" and which one "2". 
    If you swap them, the probability of ionization MUST remain exactly the same.
    
    Standard networks don't know this and might give different answers for [m1, m2] vs [m2, m1].
    This wrapper forces the network to calculate the answer for BOTH configurations
    and average them, guaranteeing that symmetry is respected mathematically.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model # The actual brain (ThreeBodyResNet)
        
    def forward(self, x, x_mirror):
        # x:        The original state [m1, m2, m3...]
        # x_mirror: The swapped state  [m2, m1, m3...] (Calculated in the Dataset class)
        
        # 1. Predict outcome for original state
        out1 = self.base(x)
        # 2. Predict outcome for swapped state
        out2 = self.base(x_mirror)
        
        # 3. Return the average. Now, f(A,B) == f(B,A) is guaranteed.
        return (out1 + out2) / 2.0

class ThreeBodyResNet(nn.Module):
    """
    The main 'Brain' of the AI. A ResNet designed for tabular physics data.
    
    It takes the 30+ physical features (Energy, Angular Momentum, Phase) and 
    maps them to a probability distribution (Bound vs Ionized).
    """
    def __init__(self, input_dim, output_dim=2, hidden_dim=512, num_layers=4, dropout_rate=0.05):
        super(ThreeBodyResNet, self).__init__()
        
        # Input Layer: Projects the raw features up to a high-dimensional space
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.BatchNorm1d(hidden_dim), 
            nn.GELU()
        )
        
        # Hidden Layers: A stack of Residual Blocks for deep reasoning
        layers = []
        for _ in range(num_layers):
            layers.append(ResidualBlock(hidden_dim, dropout_rate))
        
        # Unpack the list of layers into a Sequential module
        self.res_blocks = nn.Sequential(*layers)
        
        # Output Head: Compresses the high-dim thinking down to 2 classes (Bound/Ionized)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), 
            nn.GELU(), 
            nn.Linear(hidden_dim // 2, output_dim) # Final logits
        )
        
    def forward(self, x): 
        # Pass input through: Input Layer -> ResNet Blocks -> Output Head
        return self.output_head(self.res_blocks(self.input_layer(x)))

class FocalLoss(nn.Module):
    """
    A specialized loss function for Class Imbalance (Rare Event Detection).
    
    In the dataset, 'Ionization' events are rare. A standard loss function (CrossEntropy)
    would achieve 95% accuracy by simply guessing 'Bound' every single time.
    
    Focal Loss forces the model to focus on the 'Hard' examples (the rare Ionizations)
    by reshaping the loss curve.
    - alpha:  Weighting factor (makes rare class errors more expensive).
    - gamma:  Focusing parameter (reduces loss for easy/correct examples, magnifying hard ones).
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha     # Class weights (e.g., [1.0, 5.0])
        self.gamma = gamma     # How much to punish "easy" correct answers (Standard is 2.0)
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Calculate standard Cross Entropy loss first
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        
        # Calculate the probability of the correct class (pt = e^-loss)
        pt = torch.exp(-ce_loss)
        
        # The Focal Term: (1 - pt)^gamma
        # If the model is confident (pt -> 1), this term becomes 0. Loss disappears.
        # If the model is wrong (pt -> 0), this term becomes 1. Loss is kept high.
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean': return focal_loss.mean()
        return focal_loss.sum()

# ==========================================
# 4. OPTIMIZATION & TRAINING
# ==========================================
def run_optimization(study_name, dataset, device):
    """
    Manages Hyperparameter Tuning using Optuna.
    (Learning Rate, Batch Size, Dropout, hidden dimensions, number of layers, weight) 
    """
    # Connect to the SQLite database to store trial results (allows resuming later)
    storage = optuna.storages.RDBStorage(url=STORAGE_DB)
    
    try:
        # Try to load an existing study to continue where we left off
        # [CRITICAL CHANGE] Maximizing F1 Score (Balance) instead of just Accuracy
        study = optuna.load_study(study_name=study_name, storage=storage)
        print(f"Found existing study '{study_name}'.")
    except KeyError:
        # If not found, create a new study
        print(f"Study '{study_name}' not found. Creating new one (Maximize F1).")
        study = optuna.create_study(study_name=study_name, storage=storage, direction="maximize")

    # If N_TRIALS is 0, we skip optimization and just use the best found so far (or defaults)
    if N_TRIALS == 0:
        if len(study.trials) > 0:
            print("Skipping optimization (Using BEST params from database).")
            return study.best_params
        # Default fallback values if no database exists yet
        return {'lr': 1e-4, 'hidden_dim': 512, 'num_layers': 4, 'dropout': 0.05, 
                'batch_size': 4096, 'w_ion': WEIGHT_IONIZATION}

    def objective(trial):
        """
        The function that Optuna runs. It builds a model with random settings,
        trains it briefly, and returns the score.
        """
        # 1. Hyperparameters: Ask Optuna to suggest values from a range
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)       # Learning Rate
        drop = trial.suggest_float("dropout", 0.0, 0.4)            # Dropout %
        bs = trial.suggest_categorical("batch_size", [2048, 4096, 8192]) # Batch Size
        h_dim = trial.suggest_categorical("hidden_dim", [256, 512]) # Neurons per layer
        n_layers = trial.suggest_int("num_layers", 2, 6)           # Depth of network
        
        # [NEW] Search for best Ionization Weight (Aggressive Filtering)
        # How much more important is an Ionization event vs a Bound event?
        w_ion = trial.suggest_float("w_ion", 1.0, 100.0)

        # 2. Split Data
        # We perform a Train/Validation split just for this trial
        ds_subset = dataset
        t_size = int(0.8 * len(ds_subset)) # 80% Training
        ds_t, ds_v = random_split(ds_subset, [t_size, len(ds_subset) - t_size])
        
        # Create DataLoaders to feed the GPU
        train_loader = DataLoader(ds_t, batch_size=bs, shuffle=True, num_workers=0, drop_last=True)
        val_loader = DataLoader(ds_v, batch_size=bs, shuffle=False, num_workers=0)

        # 3. Model Setup
        # Create the Brain with the suggested dimensions
        base_model = ThreeBodyResNet(dataset.X_orig.shape[1], 2, h_dim, n_layers, drop)
        # Wrap it in the Symmetry Enforcer
        model = InvariantThreeBodyNet(base_model).to(device)
        # Optimizer (AdamW is standard for Transformers/ResNets)
        opt = optim.AdamW(model.parameters(), lr=lr)
        
        # [APPLY DYNAMIC WEIGHT]
        # Tell the Loss Function: "Pay w_ion times more attention to Class 1 (Ionization)"
        alpha = torch.tensor([1.0, w_ion]).to(device)
        crit = FocalLoss(gamma=4.0, alpha=alpha)

        # 4. Training Loop (Short run for optimization)
        for epoch in range(EPOCHS_OPT):
            model.train()
            for x, x_m, y in train_loader:
                x, x_m, y = x.to(device), x_m.to(device), y.to(device)
                opt.zero_grad()
                
                # Forward Pass (Predict)
                out = model(x, x_m)
                
                # Calculate Loss (Error)
                loss = crit(out, y.long())
                
                # Backward Pass (Learn)
                loss.backward()
                opt.step()
        
        # 5. Validation (METRIC: MACRO F1 SCORE)
        # We evaluate the model on data it hasn't seen to check for generalization
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad(): # Disable gradient calculation for speed
            for x, x_m, y in val_loader:
                x, x_m, y = x.to(device), x_m.to(device), y.to(device)
                out = model(x, x_m)
                
                # Get predictions (Highest probability wins)
                probs = F.softmax(out, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        # Calculate Macro F1: The harmonic mean of Precision and Recall.
        # This is the best metric for imbalanced data (unlike Accuracy).
        score = f1_score(all_targets, all_preds, average='macro')
        return score

    # Run the study for N_TRIALS iterations
    study.optimize(objective, n_trials=N_TRIALS)
    
    # Return the best set of parameters found
    return study.best_params

# ==========================================
# 4. TRAIN FUNCTION
# ==========================================
def train_stage1(dataset, params, device):
    """
    Executes the final training run using the best hyperparameters found by Optuna.
    """
    print(f"\n--- Training Ionization (Epochs={EPOCHS_IONIZATION}) ---")
    
    # Extract the best parameters (or use defaults if not provided)
    lr = params.get('lr', 1e-4)            # Learning Rate
    bs = params.get('batch_size', 4096)    # Batch Size
    h_dim = params.get('hidden_dim', 512)  # Network Width
    n_layers = params.get('num_layers', 4) # Network Depth
    drop = params.get('dropout', 0.05)     # Regularization
    
    # [NEW] Extract Optimized Weight
    # This value was tuned by Optuna to find the perfect balance between
    # catching all ionizations (Recall) and not making too many mistakes (Precision).
    w_ion = params.get('w_ion', 5.0)
    print(f"Using Optimized Ionization Weight: {w_ion:.2f}")

    # --- SAMPLER SETUP ---
    # Even though we use Focal Loss, physical batches usually contain 95% Bound / 5% Ionized.
    # This sampler forces every batch to contain roughly 50/50 mix.
    # Why? It prevents the model from getting bored seeing only 'Bound' cases 
    # and stabilizes the gradient updates.
    counts = np.bincount(dataset.y)
    weights = 1. / (counts + 1e-6) # Inverse frequency weights
    # WeightedRandomSampler picks samples based on these weights
    sampler = WeightedRandomSampler(weights[dataset.y], len(dataset.y), replacement=True)
    
    # Data Loader: Feeds the GPU with data
    loader = DataLoader(dataset, batch_size=bs, sampler=sampler, num_workers=4 if os.name == 'nt' else 2)
    
    # Initialize Model Architecture
    base_model = ThreeBodyResNet(dataset.X_orig.shape[1], 2, h_dim, n_layers, drop)
    # Wrap in Symmetry Enforcer
    model = InvariantThreeBodyNet(base_model).to(device)
    
    # Initialize Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # Learning Rate Scheduler: Cosine Annealing with Warm Restarts
    # This periodically raises the LR and lowers it again.
    # Why? It helps the model 'jump' out of bad local minima (ruts) and find better solutions.
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # [APPLY OPTIMIZED WEIGHT]
    # Set the cost of missing an Ionization event to 'w_ion' times higher than normal.
    alpha = torch.tensor([1.0, w_ion]).to(device)
    criterion = FocalLoss(gamma=4.0, alpha=alpha)
    
    loss_hist = []
    lr_hist = []
    
    # --- TRAINING LOOP ---
    for epoch in range(EPOCHS_IONIZATION):
        model.train() # Set model to training mode (enables Dropout/BatchNorm)
        total_loss = 0
        
        for x, x_m, y in loader:
            x, x_m, y = x.to(device), x_m.to(device), y.to(device)
            
            # 1. Clear previous gradients
            optimizer.zero_grad()
            
            # 2. Forward Pass (Make a guess)
            outputs = model(x, x_m)
            
            # 3. Calculate Loss (How bad was the guess?)
            loss = criterion(outputs, y.long())
            
            # 4. Backward Pass (Calculate corrections)
            loss.backward()
            
            # 5. Update Weights (Apply corrections)
            optimizer.step()
            
            total_loss += loss.item()
        
        # Log Statistics
        current_lr = optimizer.param_groups[0]['lr']
        loss_hist.append(total_loss/len(loader))
        lr_hist.append(current_lr)
        
        # Step the Scheduler (adjust LR for next epoch)
        scheduler.step(epoch)
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS_IONIZATION} | Loss: {loss_hist[-1]:.4f}")
            
    return model, loss_hist, lr_hist

# ==========================================
# 5. MAIN
# ==========================================
if __name__ == "__main__":
    """
    Workflow:
    1. SETUP: Detect GPU and initialize the Physics Engine.
    2. DATA PREP: Load 'Ionization' training data with augmentation (Oversampling).
    3. TUNING: Run Optuna to optimize Hyperparameters (LR, Batch Size, etc.).
    4. TRAINING: Train the final model using the best parameters and save weights.
    5. HYBRID INFERENCE: Predict outcomes using the "Physics-Informed" strategy:
         - Veto 1: Energy Conservation (Is Ionization energetically possible?)
         - Veto 2: Hyperbolic Geometry (Is it a distant, safe fly-by?)
         - AI Core: The Neural Network decides only on the remaining chaotic cases.
    6. FILTERING: Remove 'Ionized' cases (solved); save 'Bound' cases for Stage 2.
    7. VALIDATION: Generate Physics Accuracy plots, Failure Maps, and Impact Analysis.
    """
    
    # Check if a GPU is available (faster training) or fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Initialize the Physics Engine (handles all math & transformations)
    physics = ThreeBodyPhysics()
    
    # ---------------------------------------------------------
    # 1. LOAD & TRAIN
    # ---------------------------------------------------------
    print("Loading Data...")
    
    # ds_train is the actual training set. 
    # The __init__ method automatically applies the 'Chaotic Core' filter.
    # augment=True enables Oversampling to fix the class imbalance.
    ds_train = ThreeBodyDataset(TRAIN_FILE, physics, mode='ionization', augment=True)
    
    print("\n--- Hyperparameter Optimization ---")
    # run_optimization uses Optuna to test 100 different settings (trials).
    # It returns the dictionary of settings that achieved the highest F1 Score.
    best_params = run_optimization("opt_ionization_v23", ds_train, device)
    print("Best Params:", best_params)

    # Train the final production model using the best parameters found above.
    # train_stage1 runs for more epochs to ensure the model converges fully.
    model, loss_hist, lr_hist = train_stage1(ds_train, best_params, device)
    
    print(f"Saving Model to {SAVE_MODEL_FILE}...")
    # Save the learned weights to disk so we can use them later without retraining.
    torch.save(model.state_dict(), SAVE_MODEL_FILE)

    # ---------------------------------------------------------
    # 2. EVALUATE (SMART INFERENCE)
    # ---------------------------------------------------------
    print(f"\nEvaluating on {TEST_FILE}...")
    
    # Load the Test Dataset (Unseen data)
    df_test = pd.read_csv(TEST_FILE, sep=r'\s+', engine='python')
    # Ground Truth Labels (Used only for final accuracy metrics, not for prediction)
    true_labels = (df_test['OUTCOME'].astype(int) == 3).astype(int).values 
    
    # Prepare Data for AI:
    # 1. Convert physics variables to Feature Vectors (using the same scaler as training)
    X_test_orig = torch.tensor(ds_train.scaler.transform(physics.convert_batch_to_state(df_test)), dtype=torch.float32).to(device)
    
    # 2. Generate Mirror Views (Swap stars 1 & 2) for the Invariant Network
    df_mirror = df_test.copy()
    df_mirror['m1'], df_mirror['m2'] = df_test['m2'], df_test['m1']; df_mirror['psi'] += np.pi
    X_test_mirror = torch.tensor(ds_train.scaler.transform(physics.convert_batch_to_state(df_mirror)), dtype=torch.float32).to(device)
    
    # Set model to Evaluation Mode (disables Dropout for consistent predictions)
    model.eval()
    with torch.no_grad():
        # Forward pass: Feed both Original and Mirror views into the Siamese Network.
        # It outputs raw logits -> Softmax converts them to Probabilities.
        # [:, 1] extracts the probability of Class 1 (Ionization).
        probs = F.softmax(model(X_test_orig, X_test_mirror), dim=1)[:, 1].cpu().numpy()
        
    # --- SMART PREDICTION LOGIC (HYBRID PIPELINE - V2 PHYSICS) ---
    print("Applying Physics Veto (Energy & Gravitational Focusing)...")
    
    # 1. Get Physics Flag
    # Identify cases that are energetically impossible (E < 0).
    is_possible = physics.get_physics_flags(df_test)
    
    # 2. Calculate "Safe Fly-by" Filter using Hyperbolic Mechanics
    # Standard impact parameter 'b' is misleading because gravity pulls stars closer.
    # We calculate r_min (True Closest Approach) to see if a collision is actually possible.
    
    G = 4.302e-3
    M_tot = df_test['m1'] + df_test['m2'] + df_test['m3']
    v_inf = df_test['v_km_s']
    b = df_test['b_pc']
    
    # Calculate Hyperbolic Semi-Major Axis (a_hyp)
    # Measures the 'scale' of the hyperbolic trajectory.
    a_hyp = (G * M_tot) / (v_inf**2 + 1e-9)
    
    # Calculate Eccentricity of the encounter orbit
    e_hyp = np.sqrt(1 + (b / a_hyp)**2)
    
    # Calculate True Pericenter (r_min)
    # The absolute closest distance the intruder gets to the binary center of mass.
    r_min = a_hyp * (e_hyp - 1)
    
    # Logic: If the star penetrates within 5 binary radii (5a), it interacts strongly.
    # If r_min > 5a, it is a distant fly-by (tidal forces are negligible).
    is_flyby = r_min > (5.0 * df_test['a_pc'])
    
    # 3. Initialize Predictions to 0 (Bound)
    # Default assumption: The system stays together.
    preds = np.zeros(len(df_test), dtype=int)
    
    # 4. Apply Logic (The "Hybrid" Decision)
    
    # Define the "AI Zone" (The Chaos Core):
    # The AI is ONLY allowed to decide if:
    # 1. Ionization is physically possible (E_tot >= 0)
    # 2. It is NOT a safe fly-by (It is a close encounter)
    mask_ai_zone = is_possible & (~is_flyby)
    
    # For cases inside the AI Zone, use the Neural Network's probability.
    # If prob > THRESH_IONIZATION, predict Ionization (1).
    preds[mask_ai_zone] = (probs[mask_ai_zone] > THRESH_IONIZATION).astype(int)
    
    # Reporting Statistics
    n_imp = np.sum(~is_possible)            # Vetoed by Energy Conservation
    n_fly = np.sum(is_flyby & is_possible)  # Vetoed by Geometry (Too far away)
    n_ai = np.sum(mask_ai_zone)             # Handled by Neural Network
    print(f"   - Physically Impossible (Veto): {n_imp} samples")
    print(f"   - Safe Fly-bys (r_min > 5a):    {n_fly} samples")
    print(f"   - Chaotic Core (AI Decides):    {n_ai} samples")
    
    # ---------------------------------------------------------
    # 3. FILTER & SAVE DATA
    # ---------------------------------------------------------
    # We want to pass only the UNSOLVED cases to Stage 2.
    # Stage 1 detects Ionizations. If Pred == 1, the simulation is Over.
    # If Pred == 0 (Non-Ionization), the simulation continues (Exchange/Flyby).
    
    # Identify indices where the prediction is 0 (Bound)
    indices_for_stage2 = [i for i, p in enumerate(preds) if p == 0]
    
    print(f"Total Test Samples: {len(df_test)}")
    print(f"Classified as Ionization: {np.sum(preds == 1)}")
    print(f"Passed to Stage 2 (Non-Ionization): {len(indices_for_stage2)}")
    
    # Subset the dataframe and save it for the next Python script
    df_stage2 = df_test.iloc[indices_for_stage2].copy()
    df_stage2.to_csv(SAVE_DATA_FILE, sep=' ', index=False)
    print(f"Saved filtered data to '{SAVE_DATA_FILE}'")

    # ==========================================
    # 6. DIAGNOSTICS & PLOTS
    # ==========================================
    # Visualization and Quality Assurance.

    print("\n--- Running Advanced Physics Diagnostics ---")
    
    # We recalculate these physics variables on the fly for the test set.
    # This ensures the plotting logic is independent of the dataset class logic.
    m1_test = df_test['m1'].values; m2_test = df_test['m2'].values; m3_test = df_test['m3'].values
    a_test = df_test['a_pc'].values; v_inf_test = df_test['v_km_s'].values
    G = 4.302e-3
    
    # Recalculate Energy Ratios for plotting
    E_bin = -G * m1_test * m2_test / (2 * a_test)
    E_inf = 0.5 * m3_test * v_inf_test**2
    # Hardness Ratio: Kinetic Energy / Binding Energy.
    # High Hardness (>10) means the intruder is extremely fast/powerful (likely Ionization).
    hardness_ratio_val = E_inf / (np.abs(E_bin) + 1e-9)
    # Mass Ratio: How heavy is star 1 compared to star 2?
    mass_ratio_val = m1_test / (m2_test + 1e-9)

    # 1. CONFIDENCE HISTOGRAM
    # Shows how "sure" the AI is about its answers.
    # Good Model: Two peaks at 0.0 and 1.0 (Confident).
    # Bad Model: One big lump around 0.5 (Confused).
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
    # Function to slice the data into bins (e.g., Hardness 0-1, 1-10, 10-100)
    # and check the AI's accuracy in each specific slice.
    def plot_binned_accuracy(variable, var_name, y_true, y_pred, bins=10, log_scale=False):
        if log_scale:
            # Create logarithmic bins (0.1, 1, 10, 100...)
            bin_edges = np.logspace(np.log10(variable.min()+1e-9), np.log10(variable.max()), bins+1)
        else:
            # Create linear bins (0, 10, 20, 30...)
            bin_edges = np.linspace(variable.min(), variable.max(), bins+1)
            
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        accuracies = []
        counts = []
        
        for i in range(bins):
            # Isolate data points that fall into this specific bin
            mask = (variable >= bin_edges[i]) & (variable < bin_edges[i+1])
            if np.sum(mask) > 0:
                # Calculate accuracy for this specific group
                acc = np.mean(y_true[mask] == y_pred[mask])
                accuracies.append(acc)
                counts.append(np.sum(mask))
            else:
                accuracies.append(0)
                counts.append(0)
                
        plt.figure(figsize=(8, 4))
        # Plot Accuracy Line (Teal)
        plt.plot(bin_centers, accuracies, marker='o', linestyle='-', color='teal')
        plt.title(f"Accuracy vs. {var_name}")
        plt.xlabel(var_name)
        plt.ylabel("Accuracy")
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        if log_scale: plt.xscale('log')
        
        # Plot Sample Count Bar Chart (Gray) - shows where most data lies
        ax2 = plt.gca().twinx()
        ax2.bar(bin_centers, counts, width=np.diff(bin_edges), alpha=0.1, color='gray', align='edge')
        ax2.set_ylabel("Count")
        
        plt.savefig(f"stage1_acc_vs_{var_name.split()[0]}.png")
        plt.show()

    # Plot 1: Accuracy vs Hardness
    # We expect Accuracy to be high at extreme hardness (easy physics), lower in the middle.
    plot_binned_accuracy(hardness_ratio_val, "Hardness Ratio (E_kin / E_bin)", true_labels, preds, log_scale=True)
    
    # Plot 2: Accuracy vs Mass Ratio
    # We expect Accuracy to be lower when Mass Ratio ~ 1.0 (Chaotic 3-body resonance).
    plot_binned_accuracy(mass_ratio_val, "Mass Ratio (m1/m2)", true_labels, preds, log_scale=False)

    # 3. FAILURE MAP
    # A scatter plot showing EXACTLY where the AI made mistakes.
    # Red X = Wrong, Green O = Correct.
    # This helps diagnose if errors are random or clustered in a "Blind Spot".
    plt.figure(figsize=(8, 6))
    correct_mask = (true_labels == preds)
    log_hardness = np.log10(hardness_ratio_val + 1e-9)
    
    # Plot Correct Points (Green, faint)
    plt.scatter(mass_ratio_val[correct_mask], log_hardness[correct_mask], 
                c='green', s=5, alpha=0.1, label='Correct')
    # Plot Error Points (Red, visible)
    plt.scatter(mass_ratio_val[~correct_mask], log_hardness[~correct_mask], 
                c='red', marker='x', s=20, alpha=0.6, label='Wrong')
    
    plt.title("Failure Map: Mass Ratio vs Hardness")
    plt.xlabel("Mass Ratio (m1/m2)")
    plt.ylabel("Log10 Hardness Ratio")
    plt.legend()
    plt.savefig("stage1_failure_map.png")
    plt.show()

    # 4. CONFUSION MATRIX (NORMALIZED)
    # Shows percentages (e.g., "AI detected 95% of Ionizations").
    cm = confusion_matrix(true_labels, preds)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=['Bound', 'Ionized'], yticklabels=['Bound', 'Ionized'])
    plt.title('Stage 1: Ionization Recall (Smart Physics Veto)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig("stage1_confusion_matrix.png")
    plt.show()
    
    # 4a. RAW CONFUSION MATRIX (COUNTS)
    # Shows raw numbers (e.g., "AI missed 245 Ionizations").
    # Important for knowing the exact number of samples passed to Stage 2.
    cm = confusion_matrix(true_labels, preds)
    print("\nConfusion Matrix (Raw Counts):")
    print(cm)
    
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

    # 5. TRAINING LOGS
    # Plots Loss (Blue) and Learning Rate (Red) over time.
    # Ensure Loss goes down and Learning Rate anneals (goes up and down).
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='blue')
    ax1.plot(range(1, EPOCHS_IONIZATION+1), loss_hist, color='blue', label='Loss')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Learning Rate', color='red')
    ax2.plot(range(1, EPOCHS_IONIZATION+1), lr_hist, color='red', linestyle='--', label='LR')
    ax2.tick_params(axis='y', labelcolor='red')
    plt.title("Stage 1 Training Dynamics")
    plt.tight_layout()
    plt.savefig("stage1_training_log.png")
    plt.show()

    # 6. ROC CURVE
    # Measures how well the AI separates Bound from Ionized cases at ALL thresholds.
    # AUC = 1.0 is perfect. AUC = 0.5 is random guessing.
    
    # Clean probabilities: Set probability to 0.0 if physics says impossible.
    probs_clean = probs.copy()
    probs_clean[~is_possible] = 0.0
    
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
    
    # Final Summary Printout
    # Recall = Sensitivity (How many Ionizations did we catch?)
    print(f"\nFinal Ionization Recall: {cm[1,1] / (cm[1,1] + cm[1,0]):.2%}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # ==========================================
    # 7. PHYSICS FILTER VALIDATION PLOT
    # ==========================================
    """
    Validates the 'Hybrid' Logic.

    We divide the problem into 3 zones:
    1. IMPOSSIBLE (Energy Veto): Physics guarantees the answer is 0.
    2. FLY-BY (Distance Veto): Physics suggests the answer is 0.
    3. CHAOTIC CORE: Physics is silent. The AI must decide.
    
    This plot checks the accuracy in EACH zone separately.
    - Zone 1 & 2 MUST be near 100% (otherwise our physics formulas are wrong).
    - Zone 3 reflects the actual intelligence of the Neural Network.
    """
    print("\n--- Generating Physics Filter Validation Plot ---")

    # 1. Define the 3 Zones of your Hybrid Model
    
    # Zone A: Impossible (Energy < 0)
    # Logic: Physics says "Bound" (0). We check if that is correct against True Labels.
    mask_impossible = ~is_possible
    
    # Zone B: Safe Fly-bys (r_min > 5a)
    # Logic: We use the rigorous 'is_flyby' flag calculated via Hyperbolic Mechanics.
    # We recalculate it here to ensure the plotting logic is self-contained and verifiable.
    mask_flyby = is_flyby & is_possible
    
    # Zone C: Chaotic Core (The Rest)
    # Logic: This is the hard subset where we actually trusted the AI's prediction.
    mask_ai = is_possible & (~mask_flyby)
    
    # 2. Calculate Accuracy for each Zone
    # For Zone A & B, the system's "Prediction" is automatically 0. 
    # We compare 0 directly to the True Label.
    
    # Acc Impossible (Should be 100% unless data is corrupted)
    acc_imp = np.mean(true_labels[mask_impossible] == 0) if np.sum(mask_impossible) > 0 else 0.0
    
    # Acc Fly-by (CRITICAL CHECK: Is the filter too strong?)
    # If this is < 100%, it means we are filtering out real Ionizations!
    # If so, we need to relax the filter (e.g., change 5.0a to 7.0a).
    acc_fly = np.mean(true_labels[mask_flyby] == 0) if np.sum(mask_flyby) > 0 else 0.0
    
    # Acc AI (How smart is the Neural Net?)
    # Checks accuracy only on the cases the AI was asked to solve.
    acc_ai = np.mean(true_labels[mask_ai] == preds[mask_ai]) if np.sum(mask_ai) > 0 else 0.0
    
    # 3. Prepare Data for Plotting
    accuracies = [acc_imp, acc_fly, acc_ai]
    counts = [np.sum(mask_impossible), np.sum(mask_flyby), np.sum(mask_ai)]
    labels = ['Impossible\n(Energy Veto)', 'Fly-bys\n(Distance Veto)', 'Chaotic Core\n(AI Prediction)']
    colors = ['green', 'orange', 'blue'] # Green=Physics, Blue=AI
    
    # 4. Plot Bar Chart
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels, accuracies, color=colors, alpha=0.7)
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_ylim(0, 1.1) # Set limit slightly above 1.0 for text space
    ax.set_title('Hybrid Pipeline Performance\nFly-by Filter = 5.0a', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add labels on top of bars
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
    # Demonstrates the value of the Hybrid approach.
    
    print("\n--- Generating Impact Analysis Plot ---")
    
    # 1. Calculate Pure AI Accuracy (Hypothetical)
    # What if we ignored physics and let the AI guess on ALL samples?
    # We use the raw probability vs the threshold.
    preds_pure_ai = (probs > THRESH_IONIZATION).astype(int)
    acc_pure_ai = np.mean(true_labels == preds_pure_ai)
    
    # 2. Hybrid Accuracy (AI + Physics Veto)
    # 'preds' is the actual result from the pipeline (AI + Energy Check + Flyby Check)
    acc_hybrid = np.mean(true_labels == preds)
    
    print(f"Pure AI Accuracy: {acc_pure_ai:.2%}")
    print(f"Hybrid Accuracy:  {acc_hybrid:.2%}")
    print(f"Improvement:      {acc_hybrid - acc_pure_ai:+.2%}")

    # 3. Plot Comparison Side-by-Side
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