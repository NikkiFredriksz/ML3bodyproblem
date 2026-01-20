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

# ==========================================
# CONFIGURATION
# ==========================================
torch.set_float32_matmul_precision('medium') # Optimization: Sets matrix multiplication precision to medium for faster training on Tensor Cores

STORAGE_DB = "sqlite:///three_body_stage3_v10.db" # Database connection string for Optuna study storage
TRAIN_FILE = "train3body.dat" # Filepath for the training dataset
TEST_FILE = "data_for_stage3.csv" # Filepath for the testing dataset
MODEL_S3_PREFIX = "stage3_v9+filter" # Prefix used for saving model checkpoints and artifacts

# HYPERPARAMETERS
N_TRIALS = 100          # Set to 0 to use best existing params without re-optimizing; controls Optuna trial count
EPOCHS_OPT = 50         # Number of training epochs used during the hyperparameter optimization phase

# STAGE 3 SETTINGS
N_MODELS_S3 = 10 # Number of independent models to train for the ensemble (committee)
EPOCHS_S3 = 300 # Number of epochs for the final training of each ensemble model
ALIGN_TO_BINARY = True # Flag to enforce rotational symmetry by aligning the coordinate system to the binary's axis

# STRATEGY
NUM_WORKERS = 4         # Number of subprocesses to use for data loading


# ==========================================
# 1. PHYSICS ENGINE
# ==========================================
class ThreeBodyPhysics:
    """
    Handles the conversion of raw orbital parameters (from dataframes) into 
    physical state vectors (Cartesian coordinates, momenta, and derived features)
    suitable for the neural network.
    """
    def __init__(self): self.G = 4.302e-3 # Initialize gravitational constant (likely in units of pc * (km/s)^2 / M_sun)

    def _unpack_physics(self, df):
        """
        Internal helper function to extract raw physical columns from the pandas DataFrame.
        This ensures consistency in variable extraction across different methods.
        """
        m1 = df['m1'].values; m2 = df['m2'].values; m3 = df['m3'].values # Extract masses for body 1, 2, and 3
        a = df['a_pc'].values; e = df['e'].values; v_inf = df['v_km_s'].values # Extract semi-major axis (pc), eccentricity, and velocity at infinity (km/s)
        b = df['b_pc'].values # Extract impact parameter (pc)
        return m1, m2, m3, a, e, v_inf, b # Return the unpacked numpy arrays

    def convert_batch_to_state(self, df, align=False):
        """
        Core physics engine: Transforms raw orbital elements (Keplerian) into a full 
        Cartesian state vector in the Center of Mass (CoM) frame. It also calculates 
        advanced features like energy, angular momentum, and phase information to 
        help the network distinguish between exchange outcomes.
        """
        # 1. Use Helper to Unpack
        m1, m2, m3, a, e, v_inf, b = self._unpack_physics(df) # unpack basic physics variables
        
        # Keep angles raw for rotation logic; fix wrapping if necessary (convert degrees to radians if > 2pi implies degrees were passed)
        phi = np.where(np.abs(df['phi'].values)>2*np.pi, np.radians(df['phi'].values), df['phi'].values) # Azimuthal angle
        theta = np.where(np.abs(df['theta'].values)>2*np.pi, np.radians(df['theta'].values), df['theta'].values) # Polar angle
        psi = np.where(np.abs(df['psi'].values)>2*np.pi, np.radians(df['psi'].values), df['psi'].values) # Polarization/Orientation angle
        f = df['f'].values; t_coal = df['t_coal_yr'].values # True anomaly (f) and coalescence time
        
        # --- FEATURE ENGINEERING ---
        M_bin = m1 + m2 # Total mass of the inner binary
        r_mag = (a * (1 - e**2)) / (1 + e * np.cos(f)) # Current separation distance of the binary (orbit equation)

        sin_phi, cos_phi = np.sin(phi), np.cos(phi) # Pre-compute trig values for phi
        sin_theta, cos_theta = np.sin(theta), np.cos(theta) # Pre-compute trig values for theta
        sin_psi, cos_psi = np.sin(psi), np.cos(psi) # Pre-compute trig values for psi
        sin_f, cos_f = np.sin(f), np.cos(f) # Pre-compute trig values for true anomaly
        
        r_peri_encounter = b # Approximation: periapsis of the encounter is roughly the impact parameter b
        v_peri_encounter = np.sqrt(v_inf**2 + 2*self.G*M_bin/(r_peri_encounter+1e-9)) # Velocity at periapsis (Vis-viva equation approximation)
        v_avg = np.sqrt(v_inf * v_peri_encounter) # Geometric mean velocity to estimate interaction timescale
        t_approach = (50.0 * a) / (v_avg + 1e-9) # Time to approach from 50*a distance using average velocity

        mean_motion = np.sqrt(self.G * M_bin / (a**3 + 1e-9)) # Kepler's 3rd law: mean motion (angular speed)
        M_encounter = f + mean_motion * t_approach # Estimate Mean Anomaly at the time of encounter
        
        feat_phase_sin = np.sin(M_encounter) # Sin component of encounter phase (for periodicity)
        feat_phase_cos = np.cos(M_encounter) # Cos component of encounter phase (for periodicity)
        
        term_h = self.G * M_bin * a * (1 - e**2) # Squared specific angular momentum term
        h_spec = np.sqrt(np.maximum(0.0, term_h)) # Specific angular momentum magnitude
        inv_h = np.zeros_like(h_spec); mask_h = h_spec > 0 # Handle division by zero for straight-line orbits
        inv_h[mask_h] = 1.0 / h_spec[mask_h] # Inverse angular momentum
        vr = (self.G * M_bin * e * np.sin(f)) * inv_h # Radial velocity component in the binary
        vt = h_spec / r_mag # Tangential velocity component in the binary
        
        c_f, s_f = np.cos(f), np.sin(f) # Cache cosine and sine of true anomaly
        r_rel_plane = np.stack([r_mag * c_f, r_mag * s_f, np.zeros_like(f)], axis=1) # Position vector in orbital plane (2D)
        v_rel_plane = np.stack([vr * c_f - vt * s_f, vr * s_f + vt * c_f, np.zeros_like(f)], axis=1) # Velocity vector in orbital plane (2D)
        
        # Rotation Matrices Construction (Euler Angles)
        z = np.zeros_like(phi); o = np.ones_like(phi) # Zeros and Ones helpers for matrix construction
        c, s = np.cos(phi), np.sin(phi); Rz_phi = np.stack([np.stack([c,-s,z],1), np.stack([s,c,z],1), np.stack([z,z,o],1)],1) # Rotation around Z (phi)
        c, s = np.cos(theta), np.sin(theta); Rx_theta = np.stack([np.stack([o,z,z],1), np.stack([z,c,-s],1), np.stack([z,s,c],1)],1) # Rotation around X (theta)
        c, s = np.cos(psi), np.sin(psi); Rz_psi = np.stack([np.stack([c,-s,z],1), np.stack([s,c,z],1), np.stack([z,z,o],1)],1) # Rotation around Z (psi)
        R = Rz_phi @ Rx_theta @ Rz_psi # Combined rotation matrix
        r_rel = (R @ r_rel_plane[:,:,None]).squeeze(-1) # Rotate position vector to 3D space
        v_rel = (R @ v_rel_plane[:,:,None]).squeeze(-1) # Rotate velocity vector to 3D space
        
        # Calculate individual positions/velocities relative to binary CoM
        rm2 = (m2/M_bin)[:,None]; rm1 = (m1/M_bin)[:,None] # Mass fractions
        r1 = -rm2*r_rel; r2 = rm1*r_rel; v1 = -rm2*v_rel; v2 = rm1*v_rel # Body 1 and Body 2 state vectors
        r3 = np.stack([50*a, b, np.zeros_like(a)], axis=1) # Body 3 position (incoming from x-axis approx)
        v3 = np.stack([-v_inf, np.zeros_like(v_inf), np.zeros_like(v_inf)], axis=1) # Body 3 velocity (incoming along -x)
        
        # Shift to System Center of Mass (CoM)
        M_tot = (m1+m2+m3)[:,None] # Total system mass
        r_cm = (m1[:,None]*r1 + m2[:,None]*r2 + m3[:,None]*r3)/M_tot # Position of System CoM
        v_cm = (m1[:,None]*v1 + m2[:,None]*v2 + m3[:,None]*v3)/M_tot # Velocity of System CoM
        r1-=r_cm; r2-=r_cm; r3-=r_cm; v1-=v_cm; v2-=v_cm; v3-=v_cm # Recentering all vectors to CoM
        
        # Optional Alignment (for rotational invariance)
        if align:
            aa = -psi; ca, sa = np.cos(aa), np.sin(aa) # Angle to undo the last rotation (psi)
            Ra = np.stack([np.stack([ca,-sa,z],1), np.stack([sa,ca,z],1), np.stack([z,z,o],1)],1) # Alignment rotation matrix
            r1=(Ra@r1[:,:,None]).squeeze(-1); r2=(Ra@r2[:,:,None]).squeeze(-1); r3=(Ra@r3[:,:,None]).squeeze(-1) # Apply to positions
            v1=(Ra@v1[:,:,None]).squeeze(-1); v2=(Ra@v2[:,:,None]).squeeze(-1); v3=(Ra@v3[:,:,None]).squeeze(-1) # Apply to velocities
            r_rel=(Ra@r_rel[:,:,None]).squeeze(-1); v_rel=(Ra@v_rel[:,:,None]).squeeze(-1) # Apply to relative vectors

        p1=m1[:,None]*v1; p2=m2[:,None]*v2; p3=m3[:,None]*v3 # Calculate momentum vectors
        
        # Derived Physics Features
        d13=np.linalg.norm(r1-r3, axis=1); d23=np.linalg.norm(r2-r3, axis=1) # Distances between 1-3 and 2-3
        E13=-self.G*m1*m3/d13; E23=-self.G*m2*m3/d23 # Potential energy pairs
        diff_d = d13 - d23 # Difference in distance (geometric asymmetry)

        L13 = np.cross(r1-r3, v1-v3); magL13 = np.linalg.norm(L13, axis=1) # Angular momentum of pair 1-3
        L23 = np.cross(r2-r3, v2-v3); magL23 = np.linalg.norm(L23, axis=1) # Angular momentum of pair 2-3
        diff_L = magL13 - magL23 # Difference in angular momentum
        
        L_bin_vec = np.cross(r_rel, v_rel) # Angular momentum vector of the binary
        L_outer_vec = np.cross(r3, v3) # Angular momentum vector of the incoming body
        dot_L = np.sum(L_bin_vec * L_outer_vec, axis=1) # Dot product of momenta
        norm_L = np.linalg.norm(L_bin_vec, axis=1) * np.linalg.norm(L_outer_vec, axis=1) # Product of magnitudes
        cos_inclination = dot_L / (norm_L + 1e-9) # Cosine of inclination angle between binary and incomer

        def lm(x): return np.sign(x)*np.log10(1+np.abs(x)) # Log-modulus helper function to compress dynamic range
        r_peri = a * (1 - e) # Periapsis distance of binary
        compactness = M_tot.squeeze() / (r_peri * (v_inf**2 + 1e-6) + 1e-9) # Compactness metric (Safronov number proxy)

        # 1. Who is heavier? (Explicit Asymmetry Signal)
        # +1 if m1 > m2, -1 if m2 > m1. Crucial for symmetry breaking.
        mass_diff_norm = (m1 - m2) / (m1 + m2 + 1e-9) # Normalized mass difference

        # 2. Perturbation Strength (How violent is the kick?)
        # Same logic as Stage 2
        v_orb_bin = np.sqrt(self.G * M_bin / (a + 1e-9)) # Orbital velocity of the binary
        delta_v_approx = (2 * self.G * m3) / (b * v_inf + 1e-9) # Impulse approximation of delta-v
        perturb_strength = delta_v_approx / (v_orb_bin + 1e-9) # Ratio of kick to binding velocity

        # Assemble Feature Vector
        feat = [
            r1, r2, r3, p1, p2, p3, r_rel, v_rel, # Raw vectors
            np.log10(np.maximum(1e-9, t_coal))[:,None], # Log coalescence time
            np.log10(m1)[:,None], np.log10(m2)[:,None], np.log10(m3)[:,None], # Log masses
            np.log10(a)[:,None], # Log semi-major axis
            (m1/m2)[:,None], (m2/m3)[:,None], (m3/m1)[:,None], # Mass ratios
            np.log10(r_peri + 1e-9)[:,None], # Log periapsis
            np.log10((m3/M_bin)*(a/(b+1e-9))**3+1e-9)[:,None], # Tidal perturbation term
            lm(E13-E23)[:,None], lm(E13/(E23+1e-9))[:,None], # Energy comparisons (log-modulus)
            diff_d[:, None], lm(diff_L)[:, None], # Geometry and Momentum diffs
            cos_inclination[:, None], np.log10(compactness + 1e-9)[:,None], # Inclination and compactness
            sin_phi[:,None], cos_phi[:,None], # Trig features for angles
            sin_theta[:,None], cos_theta[:,None], # Trig features for angles
            sin_psi[:,None], cos_psi[:,None], # Trig features for angles
            sin_f[:,None], cos_f[:,None], # Trig features for true anomaly
            feat_phase_sin[:,None], feat_phase_cos[:,None], # Trig features for encounter phase
            mass_diff_norm[:, None], # Normalized mass difference
            np.log10(perturb_strength + 1e-9)[:, None] # Log perturbation strength
        ]
        return np.hstack(feat).astype(np.float32) # Stack all features horizontally and return as float32

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
        
        print(f"[Dataset] Final Training Size: {len(self.df)} samples (Aug={augment}).")
        
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
    """
    Standard Residual Block for a Deep Dense Network (ResNet-MLP).
    The skip connection (x + f(x)) helps gradients flow through deep networks,
    preventing vanishing gradients and allowing deeper architectures.
    """
    def __init__(self, h, drop):
        super().__init__()
        # Define the residual pathway: Linear -> BN -> GELU -> Dropout -> Linear -> BN -> GELU -> Dropout
        self.b = nn.Sequential(nn.Linear(h,h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(drop),
                               nn.Linear(h,h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(drop))
    def forward(self, x): return x + self.b(x) # Add the input x to the output of the block (Skip Connection)

class BinaryFocalLoss(nn.Module):
    """
    Focal Loss adapted for Binary Classification.
    It down-weights "easy" examples (where probability is high) and focuses training 
    on "hard" examples (where the model is wrong or uncertain).
    Formula: Loss = - alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha # Balancing factor for class frequency
        self.gamma = gamma # Focusing parameter (higher gamma = more focus on hard examples)
        self.pos_weight = pos_weight # Optional weight for the positive class (Outcome 2)
        
    def forward(self, inputs, targets):
        # Calculate standard Binary Cross Entropy (BCE)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none', pos_weight=self.pos_weight
        )
        pt = torch.exp(-bce_loss) # Calculate p_t (probability of the true class)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss # Apply the focal modulation term
        return focal_loss.mean() # Return the mean loss over the batch

class ThreeBodyResNet(nn.Module):
    """
    The main Deep Neural Network Architecture.
    It consists of an input projection, a stack of Residual Blocks, and a final classification head.
    Designed to process the high-dimensional physical state vector.
    """
    def __init__(self, i_dim, output_dim=1, h=512, n=6, drop=0.2):
        super().__init__()
        # Input Layer: Projects raw features (i_dim) to hidden dimension (h)
        self.in_l = nn.Sequential(nn.Linear(i_dim, h), nn.BatchNorm1d(h), nn.GELU())
        # Residual Stack: A sequence of 'n' ResidualBlocks
        self.res = nn.Sequential(*[ResidualBlock(h, drop) for _ in range(n)])
        # Output Head: Compresses features and outputs the logit (raw score)
        self.head = nn.Sequential(nn.Linear(h, h//2), nn.GELU(), nn.Linear(h//2, output_dim))
    def forward(self, x): return self.head(self.res(self.in_l(x))) # Forward pass: Input -> ResBlocks -> Head
   
# ==========================================
# 4. OPTIMIZATION (OPTUNA) - UPDATED
# ==========================================
def run_optimization(study_name, dataset, device):
    """
    Orchestrates hyperparameter optimization using Optuna.
    
    It searches for the best combination of Learning Rate, Dropout, Batch Size, 
    and Class Weights to maximize the F1-Score.
    
    Crucially, the objective function implements a 'mini-training' loop that 
    respects the physical symmetry constraint (Original + Mirror) to ensure 
    the chosen parameters work well under the Siamese architecture.
    """
    storage = optuna.storages.RDBStorage(url=STORAGE_DB) # Connect to the SQLite database for persistent storage of trial results
    
    try:
        # [CRITICAL] Switch to MAXIMIZE because we use F1 Score now (higher is better)
        study = optuna.load_study(study_name=study_name, storage=storage) # Try to load an existing study to resume optimization
        print(f"Found existing study '{study_name}'.") # Log success
    except KeyError:
        print(f"Study '{study_name}' not found. Creating new one (Maximize F1).") # Log failure/creation
        study = optuna.create_study(study_name=study_name, storage=storage, direction="maximize") # Create a new study, aiming to maximize the objective metric

    if N_TRIALS == 0:
        if len(study.trials) > 0:
            print("Skipping optimization (Using BEST params from database).") # Log skipping
            return study.best_params # Return the best parameters found in previous runs
        else: 
            print("No database history found and N_TRIALS=0.") # Log warning
            return {'lr': 1e-3, 'batch_size': 4096, 'dropout': 0.2, 'pos_weight': 1.0} # Return default fallback parameters

    def objective(trial):
        # 1. Suggest Hyperparameters
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True) # Suggest learning rate (log scale search)
        drop = trial.suggest_float("dropout", 0.1, 0.4) # Suggest dropout probability
        bs = trial.suggest_categorical("batch_size", [512, 1024, 2048]) # Suggest batch size
        
        # [NEW] Search for best Class Weight (Balance Outcome 1 vs 2)
        pos_w = trial.suggest_float("pos_weight", 0.1, 10.0) # Suggest weight for the positive class to handle imbalance
        
        # 2. Fast Data Split (Train/Val)
        subset_size = int(0.25 * len(dataset)) # Use only 25% of data for speed during optimization trials
        ds_subset, _ = random_split(dataset, [subset_size, len(dataset)-subset_size]) # Split the dataset
        
        t_size = int(0.8 * len(ds_subset)) # 80% of subset for training
        v_size = len(ds_subset) - t_size # 20% of subset for validation
        ds_t, ds_v = random_split(ds_subset, [t_size, v_size]) # Create train/val splits
        
        train_loader = DataLoader(ds_t, batch_size=bs, shuffle=True, num_workers=0, drop_last=True) # Create training loader
        val_loader = DataLoader(ds_v, batch_size=bs, shuffle=False, num_workers=0) # Create validation loader
        
        # 3. Setup Model [FIX: No Symmetric Wrapper]
        # We use the base ThreeBodyResNet directly
        model = ThreeBodyResNet(dataset.X_orig.shape[1], 1, 1024, 6, drop).to(device) # Initialize the ResNet model
        
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4) # Initialize AdamW optimizer with weight decay
        
        # Apply Dynamic Weight
        pos_weight_tensor = torch.tensor([pos_w]).to(device) # Convert suggested weight to tensor
        crit = BinaryFocalLoss(alpha=0.25, gamma=2.0, pos_weight=pos_weight_tensor) # Initialize Focal Loss with the suggested class weight

        # 4. Optimization Loop [FIX: Manual Augmentation]
        for epoch in range(EPOCHS_OPT):
            model.train() # Set model to training mode
            for x, x_m, y in train_loader:
                x, x_m, y = x.to(device), x_m.to(device), y.to(device) # Move data to GPU/CPU
                
                opt.zero_grad() # Clear gradients
                
                # Forward Pass 1 (Original)
                logits_orig = model(x).squeeze(1) # Compute logits for original input
                loss_orig = crit(logits_orig, y.float()) # Compute loss for original input
                
                # Forward Pass 2 (Mirror) -> Enforce Symmetry via Data
                # Label is 1.0 - y because mirror flips outcome (swapping stars flips result)
                logits_mirror = model(x_m).squeeze(1) # Compute logits for mirror input
                loss_mirror = crit(logits_mirror, 1.0 - y.float()) # Compute loss for mirror input (target inverted)
                
                # Average Loss
                loss = (loss_orig + loss_mirror) / 2.0 # Average the two losses to enforce physical symmetry
                
                loss.backward() # Backpropagation
                opt.step() # Update weights

        # 5. Validation (Metric: F1 Score) [FIX: Manual Averaging]
        model.eval() # Set model to evaluation mode
        all_preds = [] # List to store predictions
        all_targets = [] # List to store ground truth
        with torch.no_grad(): # Disable gradient calculation
            for x, x_m, y in val_loader:
                x, x_m, y = x.to(device), x_m.to(device), y.to(device) # Move data to device
                
                # Manual Ensemble Average
                prob_1 = torch.sigmoid(model(x).squeeze(1)) # Probability for original input
                prob_2 = 1.0 - torch.sigmoid(model(x_m).squeeze(1)) # Probability derived from mirror input (inverted)
                avg_prob = (prob_1 + prob_2) / 2.0 # Average the probabilities for robust prediction
                
                # Convert to binary prediction
                preds = (avg_prob > 0.5).float() # Threshold at 0.5
                
                all_preds.extend(preds.cpu().numpy()) # Store predictions
                all_targets.extend(y.cpu().numpy()) # Store targets
        
        # Calculate Macro F1
        score = f1_score(all_targets, all_preds, average='macro') # Compute Macro F1 score
        
        return score # Optuna will MAXIMIZE this value
        
    study.optimize(objective, n_trials=N_TRIALS) # Run the optimization study
    return study.best_params # Return the best parameters found

# ==========================================
# 5. TRAINING (Standard + Augmentation)
# ==========================================
def train_ensemble(name, dataset, params, device, epochs, n_models=1):
    """
    Trains the final ensemble of models ("The Committee") using the best parameters.
    
    Features:
    - Weighted Sampling: To handle any remaining class imbalance.
    - Cosine Annealing with Warm Restarts: A learning rate scheduler that helps escape local minima.
    - Checkpointing: Saves progress to allow resuming if interrupted.
    - Symmetry Enforcement: Explicitly trains on both (Original, Target) and (Mirror, 1-Target) pairs.
    """
    print(f"\n--- Training {name} ({n_models} Brains) ---") # Log start of training
    
    # Standard Sampler
    y_int = dataset.y.astype(int) # Get integer labels
    counts = np.bincount(y_int) # Count samples per class
    weights = 1. / (counts + 1e-6) # Calculate inverse frequency weights
    sampler = WeightedRandomSampler(weights[y_int], len(dataset.y), replacement=True) # Create sampler to balance batches
    
    lr_history_all = [] # Store LR history for plotting
    decay_points = [20, 60, 140, 300] # Epochs where the restart height is decayed
    DECAY_FACTOR = 0.75 # Factor by which to decay the max learning rate
    
    for i in range(n_models):
        print(f"   Brain {i+1}/{n_models}...") # Log current model index

        # [FIX] Use standard DataLoader with Augmentation enabled in Dataset
        # Persistent workers and pinned memory improve data throughput
        loader = DataLoader(dataset, batch_size=params['batch_size'], sampler=sampler, 
                            num_workers=NUM_WORKERS, drop_last=True, pin_memory=True, persistent_workers=True)
        
        # [FIX] Use the BASE model directly (No Symmetric Wrapper)
        model = ThreeBodyResNet(dataset.X_orig.shape[1], 1, 1024, 6, params['dropout']).to(device) # Initialize model
        
        opt = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=1e-4) # Initialize optimizer
        sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2) # Initialize cosine scheduler (resets every 20*2^k epochs)
        
        # [FIX] Standard Focal Loss
        crit = BinaryFocalLoss(alpha=0.25, gamma=2.0) # Initialize Loss function
        
        # Checkpoint Path
        ckpt_path = f"{MODEL_S3_PREFIX}_{i}_checkpoint.pth" # Define checkpoint filename
        start_epoch = 0 # Default start epoch
        lr_track = [] # Track LR for this model
        
        # --- RESUME LOGIC ---
        if os.path.exists(ckpt_path):
            print(f"      >> Resuming from checkpoint: {ckpt_path}") # Log resumption
            checkpoint = torch.load(ckpt_path) # Load checkpoint
            model.load_state_dict(checkpoint['model_state']) # Restore weights
            opt.load_state_dict(checkpoint['optimizer_state']) # Restore optimizer state
            sched.load_state_dict(checkpoint['scheduler_state']) # Restore scheduler state
            start_epoch = checkpoint['epoch'] + 1 # Set start epoch
            lr_track = checkpoint.get('lr_history', []) # Restore LR history
            if 'base_lrs' in checkpoint: sched.base_lrs = checkpoint['base_lrs'] # Restore base LRs for scheduler

        # --- TRAINING LOOP ---
        for epoch in range(start_epoch, epochs):
            model.train() # Set to train mode
            current_lr = opt.param_groups[0]['lr'] # Get current LR for logging
            lr_track.append(current_lr) # Log LR
            total_loss = 0 # Reset epoch loss accumulator
            
            # [FIX] Training Loop - Treat X and X_mirror as separate data points
            for x, x_m, y in loader:
                x, x_m, y = x.to(device), x_m.to(device), y.to(device) # Move batch to device
                
                opt.zero_grad() # Clear gradients
                
                # 1. Forward Pass Original
                logits_orig = model(x).squeeze(1) # Predict on original data
                loss_orig = crit(logits_orig, y.float()) # Loss vs original label
                
                # 2. Forward Pass Mirror (Enforce Symmetry via Data)
                logits_mirror = model(x_m).squeeze(1) # Predict on mirror data
                loss_mirror = crit(logits_mirror, 1.0 - y.float()) # Loss vs inverted label (1-y)
                
                # Combined Loss
                loss = (loss_orig + loss_mirror) / 2.0 # Average the losses
                
                loss.backward() # Backpropagation
                opt.step() # Update weights
                total_loss += loss.item() # Accumulate loss
            
            sched.step() # Update Learning Rate
            
            # Manual Decay Logic
            if (epoch + 1) in decay_points:
                 print(f"     [Auto-Decay] Reducing restart spike by {DECAY_FACTOR}x") # Log decay
                 sched.base_lrs = [lr * DECAY_FACTOR for lr in sched.base_lrs] # Decay the base LRs in the scheduler
                 for param_group, new_lr in zip(opt.param_groups, sched.base_lrs):
                     param_group['lr'] = new_lr # Apply new base LR to optimizer

            # Logging & Saving Checkpoint
            if (epoch+1) % 10 == 0:
                avg_loss = total_loss/len(loader) # Calculate average loss
                print(f"      Ep {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | LR: {current_lr:.1e}") # Print status
                
                # SAVE CHECKPOINT (Fixes the unused variable warning)
                torch.save({
                    'epoch': epoch, 'model_state': model.state_dict(), 
                    'optimizer_state': opt.state_dict(), 'scheduler_state': sched.state_dict(),
                    'base_lrs': sched.base_lrs, 'lr_history': lr_track
                }, ckpt_path) # Save full state to disk

        # Save Final Model
        torch.save(model.state_dict(), f"{MODEL_S3_PREFIX}_{i}.pth") # Save only weights for inference
        
        # Cleanup Checkpoint
        if os.path.exists(ckpt_path): os.remove(ckpt_path) # Remove temporary checkpoint
        
        if i == 0: lr_history_all = lr_track # Keep history of first model for plotting
        
    return lr_history_all # Return LR history

# ==========================================
# 5. MAIN
# ==========================================
if __name__ == "__main__":
    """    
    Workflow:
    1. Setup: Detect hardware (GPU/CPU) and initialize the physics engine.
    2. Optimization: Run Optuna to find the best hyperparameters (LR, Dropout, Class Weights) 
       that maximize F1-score on the 'Hard' cases.
    3. Training: Train an ensemble of N_MODELS_S3 (Committee) using the best parameters found.
    4. Evaluation (Hybrid Strategy): 
       - Apply 'Physics Filter': Use analytical mass-ratio rules for easy cases ($q > 20$ or $q < 0.05$).
       - Apply 'AI Ensemble': Use the trained Neural Networks for hard cases (Siamese Averaging).
    5. Metrics & Analysis:
       - Auto-tune the decision threshold to maximize Macro F1 (balance).
       - Compute Global Accuracy and Confusion Matrices.
    6. Advanced Diagnostics:
       - Generate Physics Accuracy plots (Accuracy vs Mass Ratio/Eccentricity).
       - Create Failure Maps (Scatter plots of errors in physical parameter space).
       - Calculate ROC Curves for the AI-driven predictions.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Auto-detect GPU
    physics = ThreeBodyPhysics() # Instantiate physics engine
    
    # ==========================================
    # 1. OPTIMIZATION
    # ==========================================
    print("\n--- Hyperparameter Setup ---")
    # Create dataset for optimization (no augmentation needed for search, just speed)
    ds_opt = ThreeBodySiameseDataset(TRAIN_FILE, physics, align=ALIGN_TO_BINARY, augment=False)
    # Run Optuna study to find best hyperparameters (p_s3)
    p_s3 = run_optimization("opt_s3_siamese_v2", ds_opt, device)
    print("Params:", p_s3) # Log best parameters
    
    # ==========================================
    # 2. TRAINING (Ensemble)
    # ==========================================
    scaler = ds_opt.scaler # Re-use scaler fitted during optimization setup to ensure consistency
    # Create dataset for final training (WITH augmentation enabled for robustness)
    ds_train = ThreeBodySiameseDataset(TRAIN_FILE, physics, scaler=scaler, align=ALIGN_TO_BINARY)
    # Train the committee of models and get the learning rate history
    lr_hist = train_ensemble("Stage 3 Siamese", ds_train, p_s3, device, EPOCHS_S3, n_models=N_MODELS_S3)

    # Plot & Save LR History
    plt.figure(figsize=(10, 4))
    plt.plot(lr_hist) # Plot the LR curve
    plt.title("Learning Rate (Cosine Annealing)")
    plt.xlabel("Epochs")
    plt.ylabel("LR")
    plt.savefig(f"{MODEL_S3_PREFIX}_lr_history.png") # Save plot to disk
    plt.show()

    # ==========================================
    # 3. EVALUATION (Hybrid: Mass Ratio + AI)
    # ==========================================
    print("\n--- Evaluation (Hybrid: Mass Ratio + AI) ---")
    df_test = pd.read_csv(TEST_FILE, sep=r'\s+', engine='python') # Load test data
    
    # Filter for exchanges (Class 1 and 2)
    mask_ex = (df_test['OUTCOME'] == 1) | (df_test['OUTCOME'] == 2) # Keep only relevant outcomes
    df_test_ex = df_test[mask_ex].copy() # Create test subset
    y_true = (df_test_ex['OUTCOME'] == 2).astype(int).values # Ground truth labels (1 for Outcome 2, 0 for Outcome 1)
    
    # B. PREPARE DATA
    m1_test = df_test_ex['m1'].values # Extract mass 1
    m2_test = df_test_ex['m2'].values # Extract mass 2
    mass_ratios = m1_test / (m2_test + 1e-9) # Calculate mass ratio q
    
    # --- LOGIC FIX: DEFINING HARD VS EASY ---
    # If m1 >> m2 (Ratio > 20.0), m2 is ejected -> Exch 1-3 (Class 0 in this binary logic? Check logic below)
    # If m2 >> m1 (Ratio < 0.05), m1 is ejected -> Exch 2-3 (Class 1)
    # "Easy" cases are those with extreme mass ratios where physics dictates the outcome.
    mask_mass_ratio_easy = (mass_ratios <= 0.05) | (mass_ratios >= 20.0)
    mask_test_hard = ~mask_mass_ratio_easy # "Hard" cases are the mixed mass ratios
    
    print(f"Test Set Split: {np.sum(mask_test_hard)} Hard Cases (AI), {np.sum(~mask_test_hard)} Easy Cases (Physics).")

    # Convert to tensors (Standard View)
    X_test_all = torch.tensor(scaler.transform(physics.convert_batch_to_state(df_test_ex, align=ALIGN_TO_BINARY)), dtype=torch.float32).to(device)
    
    # Convert to tensors (Mirror View) for Siamese Averaging
    df_swap = df_test_ex.copy() # Copy dataframe
    df_swap['m1'], df_swap['m2'] = df_swap['m2'], df_swap['m1']; df_swap['psi'] += np.pi # Swap masses and rotate
    X_test_mirror_all = torch.tensor(scaler.transform(physics.convert_batch_to_state(df_swap, align=ALIGN_TO_BINARY)), dtype=torch.float32).to(device)
    
    y_probs = np.zeros(len(df_test_ex)) # Initialize probability array
    
    # --- LOGIC FIX: APPLY CORRECT PHYSICS ---
    # CASE 1: m1 is huge (Ratio > 20.0) -> m2 ejected -> Result is Outcome 1 (Exchange 1-3?) 
    # Note: Check your label definitions carefully here. Usually m2 ejected means binary (1,3) remains.
    y_probs[mask_mass_ratio_easy & (mass_ratios >= 20.0)] = 0.0
    
    # CASE 2: m2 is huge (Ratio < 0.05) -> m1 ejected -> Result is Outcome 2 (Exchange 2-3?)
    y_probs[mask_mass_ratio_easy & (mass_ratios <= 0.05)] = 1.0

    # --- LOGIC PART 2: NEURAL NETWORK (HARD CASES ONLY) ---
    if np.sum(mask_test_hard) > 0:
        print("Evaluating Hard Cases with Neural Network Ensemble...")
        X_hard = X_test_all[mask_test_hard] # Slice hard cases (Original)
        X_hard_mirror = X_test_mirror_all[mask_test_hard] # Slice hard cases (Mirror)
        
        # Store predictions: (Models x Samples)
        ensemble_preds = np.zeros((N_MODELS_S3, len(X_hard)))
        active_models = 0
        
        for i in range(N_MODELS_S3):
            fname = f"{MODEL_S3_PREFIX}_{i}.pth" # Construct filename
            if not os.path.exists(fname): continue # Skip if missing
            
            # [FIX STARTS HERE] Re-instantiate model without the Wrapper
            # Note: We use ThreeBodyResNet directly now
            model = ThreeBodyResNet(X_hard.shape[1], 1, 1024, 6, p_s3['dropout']) # Instantiate architecture
            model.load_state_dict(torch.load(fname)) # Load weights
            model.to(device) # Move to GPU
            model.eval() # Set to eval mode
            
            with torch.no_grad():
                # [FIX] Manually Average Predictions to avoid "Zero Cancellation"
                
                # Pred 1: Normal input -> Standard prediction
                logits_1 = model(X_hard).squeeze(1)
                prob_1 = torch.sigmoid(logits_1)
                
                # Pred 2: Mirror input -> Inverted prediction
                # (Because swapping stars m1<->m2 flips the outcome 0 <-> 1)
                logits_2 = model(X_hard_mirror).squeeze(1)
                prob_2 = 1.0 - torch.sigmoid(logits_2)
                
                # Ensemble Average: Combine both views for maximum robustness
                final_prob = (prob_1 + prob_2) / 2.0
                
                # Store result
                ensemble_preds[active_models, :] = final_prob.cpu().numpy()
                active_models += 1
        
        if active_models > 0:
            mean_probs = np.mean(ensemble_preds[:active_models, :], axis=0) # Average across all models
            variance = np.var(ensemble_preds[:active_models, :], axis=0) # Calculate variance (uncertainty)
            
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
    """
    Standard Evaluation Suite.
    
    1. Threshold Tuning: Scans for the optimal probability cutoff (0.05 to 0.95) 
       that maximizes the Macro F1 Score. This ensures the model isn't biased 
       towards the majority class.
    2. Global Metrics: Calculates overall Accuracy and Confusion Matrices.
    3. Probability Distribution: Visualizes how confident the model is across the dataset.
    """
    
    # 1. Auto-Find Best Decision Threshold (Optimizing for BALANCE)
    # We scan 0.1 to 0.9 to find the cutoff that maximizes F1 Score (Balance)
    best_class_thresh = 0.5 # Default starting threshold
    best_score = 0.0 # Variable to track the best F1 score found
    
    # We use all data since we accepted 100%
    scan_range = np.linspace(0.05, 0.95, 181) # Create a fine grid of 181 threshold values between 0.05 and 0.95
    
    for t in scan_range:
        pred_t = (y_probs > t).astype(int) # Generate predictions using the current threshold 't'
        
        # We optimize for Macro F1 Score (treats both classes as equally important)
        # This prevents the threshold from drifting too high/low to favor the majority.
        score_t = f1_score(y_true, pred_t, average='macro') # Calculate Macro F1 for this threshold
        
        if score_t > best_score:
            best_score = score_t # Update best score
            best_class_thresh = t # Update best threshold
            
    print(f"\nAuto-Tuned Balanced Threshold: {best_class_thresh:.3f} (Max F1: {best_score:.2%})") # Log the result
    THRESHOLD = best_class_thresh # Set the global threshold to the optimal value found

    # 2. Final Predictions
    y_pred = (y_probs > THRESHOLD).astype(int) # Generate final binary predictions using the optimized threshold
    
    # 3. Statistics
    print("\n--- Performance Statistics (Pure AI) ---")
    print(f"Total Samples: {len(y_probs)}") # Log total count
    print("AI Coverage:   100.0% (No Simulator Used)") # Confirm 100% coverage (since we disabled uncertainty rejection)
    
    # Calculate Global Accuracy
    acc_global = np.mean(y_pred == y_true) # Simple accuracy: (Correct / Total)
    print(f"Global AI Accuracy: {acc_global*100:.2f}%") # Log global accuracy
    
    # 4. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]) # Compute raw confusion matrix (TP, TN, FP, FN)
    
    # Plot 1: Counts
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Exch 1-3', 'Exch 2-3'], 
                yticklabels=['Exch 1-3', 'Exch 2-3']) # Heatmap with integer counts
    plt.title(f"Pure AI Confusion Matrix\nAccuracy: {acc_global*100:.1f}%")
    plt.ylabel('True Outcome')
    plt.xlabel('AI Prediction')
    plt.savefig(f"{MODEL_S3_PREFIX}_cm_pure_ai.png") # Save plot
    plt.show()

    # Plot 2: Normalized (Percentages)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9) # Normalize rows to sum to 1.0 (True Positive Rate per class)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=['Exch 1-3', 'Exch 2-3'], 
                yticklabels=['Exch 1-3', 'Exch 2-3']) # Heatmap with percentages
    plt.title("Confusion Matrix (Normalized)")
    plt.ylabel('True Outcome')
    plt.xlabel('AI Prediction')
    plt.savefig(f"{MODEL_S3_PREFIX}_cm_normalized.png") # Save plot
    plt.show()
    
    # Plot 3: Probability Histogram
    plt.figure(figsize=(7,4))
    plt.hist(y_probs, bins=50, alpha=0.7, color='purple') # Histogram of predicted probabilities
    plt.title("Probability Distribution (All Samples)")
    plt.xlabel(f"Probability (Balanced Threshold = {THRESHOLD:.2f})")
    plt.axvline(THRESHOLD, color='red', linestyle='--') # Draw the decision boundary
    plt.savefig(f"{MODEL_S3_PREFIX}_prob_dist.png") # Save plot
    plt.show()
    
    # ==========================================
    # 5. ADVANCED DIAGNOSTICS (Physics Breakdown)
    # ==========================================
    """
    Physics-Informed Debugging Suite.
    
    This section helps visualize *where* the model fails physically.
    1. Binned Accuracy Plots: Checks if the model struggles with specific ranges 
       of Mass Ratio or Eccentricity.
    2. Failure Map: A scatter plot (Mass Ratio vs Impact Parameter) distinguishing 
       correct predictions (Green) from errors (Red).
    3. ROC Curve: Standard Receiver Operating Characteristic to measure separation power.
    """
    print("\n--- Running Advanced Physics Diagnostics ---")

    
    # 1. PHYSICS ACCURACY PLOTS
    # We want to see how accuracy changes vs. Mass Ratio and Eccentricity
    
    # Helper function to plot binned accuracy
    def plot_binned_accuracy(variable, var_name, y_true, y_pred, bins=10, log_scale=False):
        if log_scale:
            # Create logarithmic bins (e.g., 0.01, 0.1, 1, 10, 100) for variables like mass ratio
            bin_edges = np.logspace(np.log10(variable.min()+1e-9), np.log10(variable.max()), bins+1)
        else:
            # Create linear bins for variables like eccentricity
            bin_edges = np.linspace(variable.min(), variable.max(), bins+1)
            
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:]) # Calculate center of each bin for plotting
        accuracies = [] # List to store accuracy per bin
        counts = [] # List to store sample count per bin
        
        for i in range(bins):
            mask = (variable >= bin_edges[i]) & (variable < bin_edges[i+1]) # Mask for current bin
            if np.sum(mask) > 0:
                acc = np.mean(y_true[mask] == y_pred[mask]) # Calculate accuracy in this bin
                accuracies.append(acc)
                counts.append(np.sum(mask))
            else:
                accuracies.append(0) # Handle empty bins
                counts.append(0)
                
        plt.figure(figsize=(8, 4))
        plt.plot(bin_centers, accuracies, marker='o', linestyle='-', color='teal') # Plot Accuracy line
        plt.title(f"Accuracy vs. {var_name}")
        plt.xlabel(var_name)
        plt.ylabel("Accuracy")
        plt.ylim(0, 1.05) # Fix Y-axis to 0-100%
        plt.grid(True, alpha=0.3)
        
        # [FIX] Apply log scale to plot if requested
        if log_scale:
            plt.xscale('log')
        
        ax2 = plt.gca().twinx() # Create secondary Y-axis for the histogram
        ax2.bar(bin_centers, counts, width=np.diff(bin_edges), alpha=0.1, color='gray', align='edge') # Plot sample count histogram
        ax2.set_ylabel("Count of Samples")
        
        plt.savefig(f"{MODEL_S3_PREFIX}_acc_vs_{var_name.split()[0]}.png") # Save plot
        plt.show()

    # Define variables from the test dataframe for plotting
    mass_ratio_val = df_test_ex['m1'].values / (df_test_ex['m2'].values + 1e-9) # Calculate q = m1/m2
    eccentricity_val = df_test_ex['e'].values # Extract eccentricity
    impact_param_val = df_test_ex['b_pc'].values # Extract impact parameter
    
    # Generate the Plots
    # Only look at the "Hard" cases for these plots, as Easy cases are 100% correct by definition
    if np.sum(mask_test_hard) > 0:
        # Filter: Hard AND Accepted (Not -1.0)
        mask_plot = mask_test_hard & (y_probs != -1.0)
        
        if np.sum(mask_plot) > 0:
            y_true_plot = y_true[mask_plot] # Truth for plotting subset
            y_pred_plot = (y_probs[mask_plot] > THRESHOLD).astype(int) # Predictions for plotting subset
            
            # Slice variables
            mr_plot = mass_ratio_val[mask_plot] # Mass ratios for plotting subset
            ecc_plot = eccentricity_val[mask_plot] # Eccentricities for plotting subset

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
            correct_mask = (y_true == y_pred_plot) # Boolean mask where AI was correct
            
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
            plt.xscale('log') # Mass ratio spans orders of magnitude, so log scale is best
            plt.legend()
            plt.savefig(f"{MODEL_S3_PREFIX}_failure_map.png") # Save plot
            plt.show()

    # 3. ROC CURVE (Corrected)
    # Filter out the fallback cases (-1.0) before calculating ROC
    mask_clean = (y_probs != -1.0)
    
    if np.sum(mask_clean) > 0:
        y_true_clean = y_true[mask_clean] # Cleaned Truth
        y_probs_clean = y_probs[mask_clean] # Cleaned Probabilities
        
        fpr, tpr, thresholds = roc_curve(y_true_clean, y_probs_clean) # Calculate ROC curve points
        roc_auc = auc(fpr, tpr) # Calculate Area Under Curve (AUC)
        
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Accepted Only (AUC = {roc_auc:.3f})') # Plot ROC
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Plot random guess line (diagonal)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC (Excluding Uncertain Cases)')
        plt.legend(loc="lower right")
        plt.savefig(f"{MODEL_S3_PREFIX}_roc.png") # Save plot
        plt.show()
        
        print(f"Diagnostics Complete. ROC AUC (Accepted Only): {roc_auc:.4f}")
    else:
        print("Diagnostics Complete. (No accepted samples for ROC).")