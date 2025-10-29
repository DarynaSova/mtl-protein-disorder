import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, roc_auc_score
import joblib
from pathlib import Path

# ==== DEVICE SUPPORT ====
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
)
print(f"Using device: {device}")

# ==== LOAD DATA ====
if len(sys.argv) < 2:
    print("Error: Provide csv file. Usage: poetry run python mtl_protein_disorder/train.py yourdata.csv")
    sys.exit(1)
csv_path = Path(sys.argv[1])
df = pd.read_csv(csv_path)

# ==== FEATURES & TARGETS ====
numeric_features = ["rmsf", "bfactors", "plddt", "gscore", "pLDDT_flex"]
aa_feature = ["residue_letter"]
disprot_label = "DisProt_Label"

# Multi-task: Predict binary disorder (DisProt_Label) & continuous flexibility (rmsf)
target1 = disprot_label  # Classification
target2 = "rmsf"         # Regression/proxy for flexibility

df.dropna(subset=numeric_features + aa_feature + [target1, target2], inplace=True)

# ==== TRAIN/TEST SPLIT ====
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df[disprot_label], random_state=1)

# ==== ENCODING ====
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
aa_train = encoder.fit_transform(train_df[aa_feature])
aa_test = encoder.transform(test_df[aa_feature])

scaler = StandardScaler()
num_train = scaler.fit_transform(train_df[numeric_features])
num_test = scaler.transform(test_df[numeric_features])

X_train = np.concatenate([num_train, aa_train], axis=1)
X_test = np.concatenate([num_test, aa_test], axis=1)

# Targets
y1_train = train_df[target1].values.astype(np.float32) # binary
y2_train = train_df[target2].values.astype(np.float32) # regression
y1_test = test_df[target1].values.astype(np.float32)
y2_test = test_df[target2].values.astype(np.float32)

# ==== DATASET ====
class MTDisorderDataset(Dataset):
    def __init__(self, X, y1, y2):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y1 = torch.tensor(y1, dtype=torch.float32) # binary
        self.y2 = torch.tensor(y2, dtype=torch.float32) # regression
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y1[idx], self.y2[idx]

train_loader = DataLoader(MTDisorderDataset(X_train, y1_train, y2_train), batch_size=64, shuffle=True)
test_loader = DataLoader(MTDisorderDataset(X_test, y1_test, y2_test), batch_size=64, shuffle=False)

# ==== MODEL ====
class MultiTaskDisorderNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        # Intrinsic disorder head (binary classification)
        self.head_disprot = nn.Linear(64, 1)
        # Flexibility head (regression)
        self.head_flex = nn.Linear(64, 1)

    def forward(self, x):
        shared = self.shared(x)
        disprot_logits = self.head_disprot(shared).squeeze()
        flex_pred = self.head_flex(shared).squeeze()
        return disprot_logits, flex_pred

input_dim = X_train.shape[1]
model = MultiTaskDisorderNet(input_dim).to(device)

# ==== LOSSES & OPTIMIZER ====
loss_bce = nn.BCEWithLogitsLoss()   # for binary
loss_mse = nn.MSELoss()             # for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==== TRAINING LOOP ====
epochs = 25
for epoch in range(1, epochs + 1):
    model.train()
    loss_sum = 0
    for X_batch, y1_batch, y2_batch in train_loader:
        X_batch = X_batch.to(device)
        y1_batch = y1_batch.to(device)
        y2_batch = y2_batch.to(device)
        optimizer.zero_grad()
        disprot_logits, flex_pred = model(X_batch)
        # Multi-task: sum losses
        loss1 = loss_bce(disprot_logits, y1_batch)
        loss2 = loss_mse(flex_pred, y2_batch)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    print(f"Epoch {epoch:2d}  TotalLoss: {loss_sum / len(train_loader):.4f}")

# ==== EVALUATION ====
model.eval()
all_probs = []
all_labels = []
all_flex_true = []
all_flex_pred = []
with torch.no_grad():
    for X_batch, y1_batch, y2_batch in test_loader:
        X_batch = X_batch.to(device)
        disprot_logits, flex_pred = model(X_batch)
        probs = torch.sigmoid(disprot_logits).cpu()
        all_probs.extend(probs.tolist())
        all_labels.extend(y1_batch.tolist())
        all_flex_true.extend(y2_batch.tolist())
        all_flex_pred.extend(flex_pred.cpu().tolist())

# Binary metrics
all_probs = np.array(all_probs)
all_labels = np.array(all_labels)
y_pred = (all_probs >= 0.5).astype(int)
print("\n--- Test Set Performance ---")
print(f"Disorder Accuracy: {accuracy_score(all_labels, y_pred):.3f}")
print(f"Disorder F1:      {f1_score(all_labels, y_pred):.3f}")
print(f"Disorder ROC-AUC: {roc_auc_score(all_labels, all_probs):.3f}")
# Regression metrics
print(f"Flexibility RMSE: {np.sqrt(mean_squared_error(all_flex_true, all_flex_pred)):.3f}")

# ==== SAVE ====
torch.save(model.state_dict(), "mt_disorder_model.pth")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoder, "encoder.pkl")
print("\nModel, scaler, and encoder saved.")




