import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import os
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

class FootballPredictor:
    def __init__(self, model_path: str = "torch_ht2ft.pt"):
        self.model_path = model_path
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.target_names = ["ft_goals_home", "ft_goals_away", "ft_sot_home", "ft_sot_away"]
        
        # Feature definitions
        self.numerical_features = [
            "ht_goals_home", "ht_goals_away", "ht_sot_home", "ht_sot_away",
            "ht_reds_home", "ht_reds_away", "elo_home", "elo_away"
        ]
        self.categorical_features = ["league", "season", "venue"]
        
    def load_and_preprocess_data(self, csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and preprocess the dataset with improved validation"""
        print(f"Loading data from {csv_path}...")
        
        # Load with better error handling
        try:
            df = pd.read_csv(csv_path, parse_dates=["date"])
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        except Exception as e:
            raise Exception(f"Error loading CSV: {e}")
            
        # Data validation
        required_cols = self.numerical_features + self.categorical_features + self.target_names + ["date"]
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        print(f"Loaded {len(df)} matches from {df['date'].min()} to {df['date'].max()}")
        
        # Handle missing values more robustly
        print("Handling missing values...")
        for col in self.numerical_features:
            if df[col].isnull().sum() > 0:
                print(f"  Filling {df[col].isnull().sum()} missing values in {col} with median")
                df[col] = df[col].fillna(df[col].median())
                
        for col in self.categorical_features:
            if df[col].isnull().sum() > 0:
                print(f"  Filling {df[col].isnull().sum()} missing values in {col} with 'Unknown'")
                df[col] = df[col].fillna('Unknown')
        
        # Sort by date and reset index
        df = df.sort_values("date").reset_index(drop=True)
        
        # Time-aware split (80-20)
        cut_idx = int(0.8 * len(df))
        df_train = df.iloc[:cut_idx].copy()
        df_test = df.iloc[cut_idx:].copy()
        
        print(f"Train set: {len(df_train)} matches (until {df_train['date'].max()})")
        print(f"Test set: {len(df_test)} matches (from {df_test['date'].min()})")
        
        return df_train, df_test
    
    def create_preprocessor(self, df_train: pd.DataFrame) -> ColumnTransformer:
        """Create preprocessing pipeline with better handling"""
        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), self.numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.categorical_features)
        ], remainder='drop')
        
        return preprocessor
    
    def prepare_data(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training and test data"""
        # Create and fit preprocessor
        self.preprocessor = self.create_preprocessor(df_train)
        
        # Transform features
        X_train = self.preprocessor.fit_transform(df_train[self.numerical_features + self.categorical_features])
        X_test = self.preprocessor.transform(df_test[self.numerical_features + self.categorical_features])
        
        # Extract targets
        y_train = df_train[self.target_names].values.astype(np.float32)
        y_test = df_test[self.target_names].values.astype(np.float32)
        
        # Store feature names for later analysis
        num_feature_names = self.numerical_features
        cat_feature_names = []
        if hasattr(self.preprocessor.named_transformers_['cat'], 'get_feature_names_out'):
            cat_feature_names = list(self.preprocessor.named_transformers_['cat'].get_feature_names_out(self.categorical_features))
        
        self.feature_names = num_feature_names + cat_feature_names
        
        print(f"Feature matrix shape: {X_train.shape}")
        print(f"Target matrix shape: {y_train.shape}")
        
        return X_train, X_test, y_train, y_test

class TabDataset(Dataset):
    """Improved Dataset class with validation"""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert len(X) == len(y), f"Feature and target lengths don't match: {len(X)} vs {len(y)}"
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self) -> int:
        return self.X.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

class ImprovedMLP(nn.Module):
    """Improved MLP with better architecture and residual connections"""
    def __init__(self, in_dim: int, out_dim: int = 4, hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
            
        layers = []
        prev_dim = in_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3 if i == 0 else 0.2)  # Higher dropout for first layer
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, out_dim))
        self.net = nn.Sequential(*layers)
        
        # Use LeakyReLU instead of Softplus for better gradients
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        # Ensure non-negativity (goals/shots can't be negative)
        return torch.clamp(raw, min=0)  # Simple clamp instead of softplus

class Trainer:
    """Training class with improved logging and validation"""
    def __init__(self, model: nn.Module, train_dl: DataLoader, test_dl: DataLoader, 
                 lr: float = 1e-3, weight_decay: float = 1e-5):
        self.model = model
        self.train_dl = train_dl
        self.test_dl = test_dl
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=15
        )
        
        # Use MSE loss as it's more suitable for regression
        self.loss_fn = nn.MSELoss()
        
        # Training history
        self.history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
        
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for X_batch, y_batch in self.train_dl:
            # Forward pass
            pred = self.model(X_batch)
            loss = self.loss_fn(pred, y_batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / num_batches
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in self.test_dl:
                pred = self.model(X_batch)
                loss = self.loss_fn(pred, y_batch)
                
                total_loss += loss.item()
                all_preds.append(pred)
                all_targets.append(y_batch)
        
        # Calculate metrics
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        val_loss = total_loss / len(self.test_dl)
        mae = (all_preds - all_targets).abs().mean().item()
        
        return val_loss, mae
    
    def train(self, epochs: int = 200, patience: int = 25, save_path: str = "best_model.pt") -> Dict[str, Any]:
        """Train the model with improved early stopping"""
        best_mae = float('inf')
        best_epoch = 0
        epochs_without_improvement = 0
        
        print("Starting training...")
        print("-" * 60)
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch()
            
            # Validation
            val_loss, val_mae = self.validate()
            
            # Learning rate scheduling
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_mae'].append(val_mae)
            
            # Print progress
            if epoch % 10 == 0 or epoch < 10:
                print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f}")
            
            # Early stopping and model saving
            if val_mae < best_mae - 1e-5:
                best_mae = val_mae
                best_epoch = epoch
                epochs_without_improvement = 0
                
                # Save best model
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'epoch': epoch,
                    'best_mae': best_mae,
                    'history': self.history
                }, save_path)
                
            else:
                epochs_without_improvement += 1
                
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                print(f"Best MAE: {best_mae:.4f} at epoch {best_epoch}")
                break
        
        return {
            'best_mae': best_mae,
            'best_epoch': best_epoch,
            'final_epoch': epoch,
            'history': self.history
        }

def evaluate_model(model: nn.Module, test_dl: DataLoader, target_names: list) -> Dict[str, float]:
    """Comprehensive model evaluation"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_dl:
            pred = model(X_batch)
            all_preds.append(pred)
            all_targets.append(y_batch)
    
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    
    # Calculate metrics for each target
    metrics = {}
    for i, target_name in enumerate(target_names):
        pred_i = all_preds[:, i]
        true_i = all_targets[:, i]
        
        metrics[f'{target_name}_mae'] = mean_absolute_error(true_i, pred_i)
        metrics[f'{target_name}_rmse'] = np.sqrt(mean_squared_error(true_i, pred_i))
        metrics[f'{target_name}_mape'] = np.mean(np.abs((true_i - pred_i) / (true_i + 1e-8))) * 100
    
    # Overall metrics
    metrics['overall_mae'] = mean_absolute_error(all_targets.flatten(), all_preds.flatten())
    metrics['overall_rmse'] = np.sqrt(mean_squared_error(all_targets.flatten(), all_preds.flatten()))
    
    return metrics

# Main execution
def main():
    # Initialize predictor
    predictor = FootballPredictor()
    
    # Load and preprocess data
    df_train, df_test = predictor.load_and_preprocess_data("matches.csv")
    X_train, X_test, y_train, y_test = predictor.prepare_data(df_train, df_test)
    
    # Create datasets and dataloaders
    train_ds = TabDataset(X_train, y_train)
    test_ds = TabDataset(X_test, y_test)
    
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)
    
    # Create model
    model = ImprovedMLP(in_dim=X_train.shape[1], out_dim=len(predictor.target_names))
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    trainer = Trainer(model, train_dl, test_dl, lr=1e-3, weight_decay=1e-4)
    results = trainer.train(epochs=200, patience=30, save_path=predictor.model_path)
    
    # Load best model and evaluate
    checkpoint = torch.load(predictor.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    metrics = evaluate_model(model, test_dl, predictor.target_names)
    
    print("\n" + "="*60)
    print("FINAL EVALUATION RESULTS")
    print("="*60)
    for metric_name, value in metrics.items():
        print(f"{metric_name:25s}: {value:.4f}")
    
    # Save preprocessor separately for inference
    with open('preprocessor.pkl', 'wb') as f:
        pickle.dump(predictor.preprocessor, f)
    
    print(f"\nModel saved to: {predictor.model_path}")
    print("Preprocessor saved to: preprocessor.pkl")

if __name__ == "__main__":
    main()
