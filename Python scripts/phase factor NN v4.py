import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import os
import tempfile

# ==========================================
# 1. PHYSICS ENGINE
# ==========================================

class ThreeBodyPhysics:
    def __init__(self):
        # Gravitational Constant in units: pc * (km/s)^2 / M_sun
        # G ~ 4.302e-3
        self.G = 4.302e-3 

    def rotation_matrix(self, phi, theta, psi):
        # Z-X-Z Euler Rotation
        Rz_phi = np.array([
            [np.cos(phi), -np.sin(phi), 0],
            [np.sin(phi),  np.cos(phi), 0],
            [0, 0, 1]
        ])
        Rx_theta = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta),  np.cos(theta)]
        ])
        Rz_psi = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi),  np.cos(psi), 0],
            [0, 0, 1]
        ])
        return Rz_phi @ Rx_theta @ Rz_psi

    def convert_row_to_state(self, row):
        # 1. Unpack Parameters
        m1, m2, m3 = row['m1'], row['m2'], row['m3']
        a, e, b = row['a_pc'], row['e'], row['b_pc']
        phi, theta, psi = row['phi'], row['theta'], row['psi']
        f, v_inf = row['f'], row['v_km_s'] 
        t_coal = row['t_coal_yr']
        
        # 2. Setup Binary System (Bodies 1 and 2)
        M_bin = m1 + m2
        # Orbital distance (polar equation of ellipse)
        r_mag = (a * (1 - e**2)) / (1 + e * np.cos(f))
        
        # Specific Angular Momentum h = sqrt(G * M * a * (1-e^2))
        h_spec = np.sqrt(max(0.0, self.G * M_bin * a * (1 - e**2))) 
        
        # Orbital Velocity Components (Vis-viva decomposition)
        if h_spec == 0: 
            vr, vt = 0.0, 0.0
        else:
             vr = (self.G * M_bin * e * np.sin(f)) / h_spec
             vt = h_spec / r_mag
             
        # Relative Position & Velocity Vectors (2D Plane)
        r_rel = np.array([r_mag * np.cos(f), r_mag * np.sin(f), 0.0])
        v_rel = np.array([vr * np.cos(f) - vt * np.sin(f), 
                          vr * np.sin(f) + vt * np.cos(f), 0.0])

        # Rotate to 3D Space
        R = self.rotation_matrix(phi, theta, psi)
        r_rel = R @ r_rel
        v_rel = R @ v_rel

        # Convert to Body 1 and Body 2 (Binary Center of Mass frame)
        r1 = -(m2 / M_bin) * r_rel
        r2 = (m1 / M_bin) * r_rel
        v1 = -(m2 / M_bin) * v_rel
        v2 = (m1 / M_bin) * v_rel

        # 3. Setup Incoming Body 3
        # Starts at x = +50a, y = b (impact param)
        r3 = np.array([50 * a, b, 0.0]) 
        v3 = np.array([-v_inf, 0.0, 0.0])
        
        # 4. Shift to Global Center of Mass
        M_tot = m1 + m2 + m3
        r_cm = (m1*r1 + m2*r2 + m3*r3) / M_tot
        v_cm = (m1*v1 + m2*v2 + m3*v3) / M_tot
        
        r1 -= r_cm; r2 -= r_cm; r3 -= r_cm
        v1 -= v_cm; v2 -= v_cm; v3 -= v_cm
        
        # Calculate Momenta
        p1 = m1 * v1; p2 = m2 * v2; p3 = m3 * v3

        # 5. Feature Engineering
        # Kinetic Energy
        T = 0.5 * (m1 * np.linalg.norm(v1)**2 + 
                   m2 * np.linalg.norm(v2)**2 + 
                   m3 * np.linalg.norm(v3)**2)
        
        # Potential Energy
        d12 = np.linalg.norm(r1 - r2)
        d13 = np.linalg.norm(r1 - r3)
        d23 = np.linalg.norm(r2 - r3)
        U = -self.G * ((m1*m2)/d12 + (m1*m3)/d13 + (m2*m3)/d23)
        
        Total_Energy = T + U
        L_vec = np.cross(r1, p1) + np.cross(r2, p2) + np.cross(r3, p3)
        
        # Log-Modulus Transform (Handles vast orders of magnitude)
        def log_modulus(x):
            return np.sign(x) * np.log10(1 + np.abs(x))

        # 6. Final State Vector (23 Dimensions)
        state = np.concatenate([
            r1, r2, r3, p1, p2, p3,                 # 18 Phase Space Coords
            [log_modulus(Total_Energy)],            # 1 Energy
            [log_modulus(val) for val in L_vec],    # 3 Angular Momentum
            [np.log10(max(1e-9, t_coal))]           # 1 Time to Coalescence
        ])
        return state

# ==========================================
# 2. DATASET HANDLING
# ==========================================

class CelestialDataset(Dataset):
    def __init__(self, filepath, physics_engine, scaler=None, is_train=True):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        # Load Data
        self.data = pd.read_csv(filepath, sep=r'\s+', engine='python')
        
        self.feature_cols = ['m1', 'm2', 'm3', 'a_pc', 'e', 'b_pc', 
                             'phi', 'theta', 'psi', 'f', 'v_km_s', 
                             'Ecc_Anomaly', 't_coal_yr']
        self.label_col = 'OUTCOME'
        
        print(f"Converting {filepath} (G={physics_engine.G})...")
        
        # Process Physics (Vectorizing this is hard due to rotation logic, list comp is fine)
        self.X_phase_space = []
        for _, row in self.data[self.feature_cols].iterrows():
            self.X_phase_space.append(physics_engine.convert_row_to_state(row))
        
        self.X_phase_space = np.array(self.X_phase_space, dtype=np.float32)
        
        # SAFETY FIX: Ensure labels are integers
        self.y = self.data[self.label_col].astype(int).values
        
        # Scaling
        if is_train:
            self.scaler = StandardScaler()
            self.X_phase_space = self.scaler.fit_transform(self.X_phase_space)
        elif scaler is not None:
            self.scaler = scaler
            self.X_phase_space = self.scaler.transform(self.X_phase_space)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.X_phase_space[idx]), torch.tensor(self.y[idx], dtype=torch.long)

# ==========================================
# 3. MODEL (ResNet)
# ==========================================

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return x + self.block(x)

class ThreeBodyResNet(nn.Module):
    def __init__(self, input_dim=23, output_dim=4):
        super(ThreeBodyResNet, self).__init__()
        
        # Parameters (Can be tuned via Optuna)
        hidden_dim = 512
        dropout_rate = 0.05
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim, dropout_rate),
            ResidualBlock(hidden_dim, dropout_rate),
            ResidualBlock(hidden_dim, dropout_rate),
            ResidualBlock(hidden_dim, dropout_rate)
        )
        
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.output_head(self.res_blocks(self.input_layer(x)))

# ==========================================
# 4. TRAINING LOOP
# ==========================================

def train_model():
    # --- HYPERPARAMETERS ---
    BATCH_SIZE = 512
    EPOCHS = 60
    LEARNING_RATE = 0.002
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Prepare Data
    physics = ThreeBodyPhysics()
    train_dataset = CelestialDataset('train3body.dat', physics, is_train=True)
    test_dataset = CelestialDataset('test3body.dat', physics, scaler=train_dataset.scaler, is_train=False)
    
    # Windows Safety: Default to 0 workers to prevent BrokenPipeError. 
    # If on Linux/Mac, or if Windows behaves, set to 4 for speed.
    workers = 0 if os.name == 'nt' else 4
    
    # Pin Memory speeds up CPU->GPU transfer
    use_pin_memory = True if device.type == 'cuda' else False
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        pin_memory=use_pin_memory, num_workers=workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        pin_memory=use_pin_memory, num_workers=workers
    )

    # 2. Initialize Model
    model = ThreeBodyResNet(input_dim=23, output_dim=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Scheduler: Restarts LR every 10 epochs
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    # 3. Handle Class Imbalance
    print("Calculating Class Weights...")
    counts = np.bincount(train_dataset.y)
    # SAFETY FIX: Add epsilon (1e-6) to prevent divide-by-zero if a class is missing
    weights = len(train_dataset.y) / (len(counts) * counts + 1e-6)
    
    class_weights = torch.FloatTensor(weights).to(device)
    print(f"Weights applied: {weights}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    # 4. Training Loop
    history = {'epoch': [], 'loss': [], 'test_acc': [], 'lr': []}
    print("\nStarting Training (Final Golden Version)...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        # Train
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            # Update LR per batch for smooth cosine curve
            scheduler.step(epoch + batch_idx / len(train_loader)) 
            
        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_acc = correct / total
        avg_loss = running_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log
        history['epoch'].append(epoch + 1)
        history['loss'].append(avg_loss)
        history['test_acc'].append(test_acc)
        history['lr'].append(current_lr)
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Test Acc: {test_acc:.4f} | LR: {current_lr:.6f}")

    # 5. Save Results
    pd.DataFrame(history).to_csv('training_log.csv', index=False)
    print("\nTraining log saved to 'training_log.csv'")
    
    save_filename = "three_body_resnet_final.pth"
    try:
        with open(os.path.abspath(save_filename), 'wb') as f:
            torch.save(model.state_dict(), f)
        print(f"Model saved successfully to: {os.path.abspath(save_filename)}")
    except Exception as e:
        print(f"Saving to temp folder due to error: {e}")
        temp_path = os.path.join(tempfile.gettempdir(), save_filename)
        with open(temp_path, 'wb') as f:
            torch.save(model.state_dict(), f)

if __name__ == "__main__":
    train_model()