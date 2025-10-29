import sys
import pandas as pd
import numpy as np
import torch
import joblib
import torch.nn as nn

# ==== DEVICE SUPPORT ====
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
)
print(f"Using device: {device}")

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
        self.head_disprot = nn.Linear(64, 1)   # Intrinsic disorder head (binary classification)
        self.head_flex = nn.Linear(64, 1)      # Flexibility head (regression)

    def forward(self, x):
        shared = self.shared(x)
        disprot_logits = self.head_disprot(shared).squeeze()
        flex_pred = self.head_flex(shared).squeeze()
        return disprot_logits, flex_pred

if len(sys.argv) < 2:
    print("Usage: poetry run python mtl_protein_disorder/inference.py your_new_data.csv")
    sys.exit(1)

# Load model/scaler/encoder --- You MUST fit with the same input_dim as training
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

# --- Load new prediction data ---
filename = sys.argv[1]
df = pd.read_csv(filename)
numeric_features = ["rmsf", "bfactors", "plddt", "gscore", "pLDDT_flex"]
aa_feature = ["residue_letter"]

# Preprocess features
aa_encoded = encoder.transform(df[aa_feature])
num_scaled = scaler.transform(df[numeric_features])
X = np.concatenate([num_scaled, aa_encoded], axis=1)
input_dim = X.shape[1]

model = MultiTaskDisorderNet(input_dim)
model.load_state_dict(torch.load("mt_disorder_model.pth", map_location=device))
model.eval()
model.to(device)

X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

# Predict
with torch.no_grad():
    disprot_logits, flex_pred = model(X_tensor)
    disprot_probs = torch.sigmoid(disprot_logits).cpu().numpy()
    flex_pred = flex_pred.cpu().numpy()

df["DisProt_PredProb"] = disprot_probs
df["Flex_Pred"] = flex_pred
df.to_csv("data/inference_results.csv", index=False)
print("Saved predictions to data/inference_results.csv")
