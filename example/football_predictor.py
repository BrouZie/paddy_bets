import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
from typing import Dict, List, Union, Optional
import warnings
warnings.filterwarnings('ignore')

class ImprovedMLP(nn.Module):
    """Same model architecture as in training script"""
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
                nn.Dropout(0.3 if i == 0 else 0.2)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, out_dim))
        self.net = nn.Sequential(*layers)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        return torch.clamp(raw, min=0)

class FootballMatchPredictor:
    """Predictor for individual football matches"""
    
    def __init__(self, model_path: str = "torch_ht2ft.pt", preprocessor_path: str = "preprocessor.pkl"):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model = None
        self.preprocessor = None
        self.target_names = ["ft_goals_home", "ft_goals_away", "ft_sot_home", "ft_sot_away"]
        self.feature_names = [
            "ht_goals_home", "ht_goals_away", "ht_sot_home", "ht_sot_away",
            "ht_reds_home", "ht_reds_away", "elo_home", "elo_away",
            "league", "season", "venue"
        ]
        self.load_model()
    
    def load_model(self):
        """Load the trained model and preprocessor"""
        try:
            # Load model
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # Load preprocessor to get input dimensions
            with open(self.preprocessor_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
            
            # Create model with correct input dimension
            # Get a dummy transform to determine input size
            dummy_data = pd.DataFrame({
                'ht_goals_home': [0], 'ht_goals_away': [0], 'ht_sot_home': [0], 'ht_sot_away': [0],
                'ht_reds_home': [0], 'ht_reds_away': [0], 'elo_home': [1500], 'elo_away': [1500],
                'league': ['Premier League'], 'season': ['2023-24'], 'venue': ['Home']
            })
            dummy_transformed = self.preprocessor.transform(dummy_data)
            input_dim = dummy_transformed.shape[1]
            
            # Initialize model
            self.model = ImprovedMLP(in_dim=input_dim, out_dim=len(self.target_names))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print(f"✅ Model loaded successfully from {self.model_path}")
            print(f"✅ Preprocessor loaded from {self.preprocessor_path}")
            print(f"Model trained for {checkpoint.get('epoch', 'unknown')} epochs with MAE: {checkpoint.get('best_mae', 'unknown'):.4f}")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model files not found. Please train the model first. Error: {e}")
        except Exception as e:
            raise Exception(f"Error loading model: {e}")
    
    def predict_single_match(self, match_data: Dict[str, Union[int, float, str]]) -> Dict[str, float]:
        """
        Predict full-time statistics for a single match
        
        Args:
            match_data: Dictionary containing match features
            
        Returns:
            Dictionary with predicted full-time statistics
        """
        # Validate input
        self._validate_match_data(match_data)
        
        # Convert to DataFrame
        df = pd.DataFrame([match_data])
        
        # Transform features
        X = self.preprocessor.transform(df[self.feature_names])
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(X_tensor).numpy()[0]
        
        # Format results
        results = {}
        for i, target in enumerate(self.target_names):
            results[target] = round(float(prediction[i]), 2)
        
        return results
    
    def predict_multiple_matches(self, matches_data: List[Dict[str, Union[int, float, str]]]) -> List[Dict[str, float]]:
        """Predict multiple matches at once"""
        results = []
        for match_data in matches_data:
            try:
                prediction = self.predict_single_match(match_data)
                results.append(prediction)
            except Exception as e:
                print(f"Error predicting match {match_data}: {e}")
                results.append(None)
        return results
    
    def predict_with_confidence(self, match_data: Dict[str, Union[int, float, str]], 
                              n_samples: int = 100) -> Dict[str, Dict[str, float]]:
        """
        Predict with confidence intervals using Monte Carlo dropout
        
        Args:
            match_data: Match features
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Dictionary with mean, std, and confidence intervals
        """
        self._validate_match_data(match_data)
        
        # Convert to DataFrame and transform
        df = pd.DataFrame([match_data])
        X = self.preprocessor.transform(df[self.feature_names])
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # Enable only dropout, keep BatchNorm in eval mode
        def enable_dropout(m):
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
        
        predictions = []
        
        # Set model to eval mode first (keeps BatchNorm in eval)
        self.model.eval()
        # Then enable only dropout layers
        self.model.apply(enable_dropout)
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.model(X_tensor).numpy()[0]
                predictions.append(pred)
        
        # Back to full eval mode
        self.model.eval()
        
        predictions = np.array(predictions)
        
        # Calculate statistics
        results = {}
        for i, target in enumerate(self.target_names):
            pred_values = predictions[:, i]
            results[target] = {
                'mean': round(float(np.mean(pred_values)), 2),
                'std': round(float(np.std(pred_values)), 2),
                'ci_lower': round(float(np.percentile(pred_values, 2.5)), 2),
                'ci_upper': round(float(np.percentile(pred_values, 97.5)), 2),
                'min': round(float(np.min(pred_values)), 2),
                'max': round(float(np.max(pred_values)), 2)
            }
        
        return results
    
    def _validate_match_data(self, match_data: Dict[str, Union[int, float, str]]):
        """Validate input match data"""
        required_features = set(self.feature_names)
        provided_features = set(match_data.keys())
        
        missing_features = required_features - provided_features
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Type validation
        numeric_features = ["ht_goals_home", "ht_goals_away", "ht_sot_home", "ht_sot_away",
                           "ht_reds_home", "ht_reds_away", "elo_home", "elo_away"]
        
        for feature in numeric_features:
            if not isinstance(match_data[feature], (int, float)):
                try:
                    match_data[feature] = float(match_data[feature])
                except ValueError:
                    raise ValueError(f"Feature '{feature}' must be numeric, got: {match_data[feature]}")
        
        # Range validation
        if match_data["ht_goals_home"] < 0 or match_data["ht_goals_away"] < 0:
            raise ValueError("Goals cannot be negative")
        if match_data["ht_sot_home"] < 0 or match_data["ht_sot_away"] < 0:
            raise ValueError("Shots on target cannot be negative")
        if match_data["ht_reds_home"] < 0 or match_data["ht_reds_away"] < 0:
            raise ValueError("Red cards cannot be negative")
    
    def get_feature_importance_estimate(self, match_data: Dict[str, Union[int, float, str]], 
                                      target_idx: int = 0) -> Dict[str, float]:
        """
        Simple feature importance estimation using perturbation
        
        Args:
            match_data: Base match data
            target_idx: Which target to analyze (0=home goals, 1=away goals, etc.)
        """
        base_prediction = self.predict_single_match(match_data)
        base_value = list(base_prediction.values())[target_idx]
        
        importance = {}
        numeric_features = ["ht_goals_home", "ht_goals_away", "ht_sot_home", "ht_sot_away",
                           "ht_reds_home", "ht_reds_away", "elo_home", "elo_away"]
        
        for feature in numeric_features:
            # Perturb feature by 10%
            perturbed_data = match_data.copy()
            original_value = perturbed_data[feature]
            perturbation = max(0.1, abs(original_value * 0.1))  # At least 0.1 change
            
            perturbed_data[feature] = original_value + perturbation
            perturbed_prediction = self.predict_single_match(perturbed_data)
            perturbed_value = list(perturbed_prediction.values())[target_idx]
            
            # Calculate sensitivity (change in output / change in input)
            sensitivity = abs(perturbed_value - base_value) / perturbation
            importance[feature] = round(sensitivity, 4)
        
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

def create_example_matches() -> List[Dict[str, Union[int, float, str]]]:
    """Create some example matches for testing"""
    return [
        {
            # Close match at half-time
            "ht_goals_home": 1,
            "ht_goals_away": 1,
            "ht_sot_home": 3,
            "ht_sot_away": 2,
            "ht_reds_home": 0,
            "ht_reds_away": 0,
            "elo_home": 1650,  # Strong home team
            "elo_away": 1580,  # Good away team
            "league": "Premier League",
            "season": "2023-24",
            "venue": "Home"
        },
        {
            # Home team dominating
            "ht_goals_home": 2,
            "ht_goals_away": 0,
            "ht_sot_home": 5,
            "ht_sot_away": 1,
            "ht_reds_home": 0,
            "ht_reds_away": 1,  # Away team has red card
            "elo_home": 1750,  # Very strong home team
            "elo_away": 1400,  # Weaker away team
            "league": "La Liga",
            "season": "2023-24",
            "venue": "Home"
        },
        {
            # Low-scoring defensive game
            "ht_goals_home": 0,
            "ht_goals_away": 0,
            "ht_sot_home": 1,
            "ht_sot_away": 1,
            "ht_reds_home": 0,
            "ht_reds_away": 0,
            "elo_home": 1520,
            "elo_away": 1510,
            "league": "Serie A",
            "season": "2023-24",
            "venue": "Home"
        }
    ]

# Example usage and testing
def main():
    print("🏈 Football Match Predictor Demo")
    print("=" * 50)
    
    try:
        # Initialize predictor
        predictor = FootballMatchPredictor()
        
        # Get example matches
        example_matches = create_example_matches()
        
        for i, match in enumerate(example_matches, 1):
            print(f"\n📊 MATCH {i} PREDICTION")
            print("-" * 30)
            
            # Show input
            print("Half-time situation:")
            print(f"  Score: {match['ht_goals_home']}-{match['ht_goals_away']}")
            print(f"  Shots on target: {match['ht_sot_home']}-{match['ht_sot_away']}")
            print(f"  Red cards: {match['ht_reds_home']}-{match['ht_reds_away']}")
            print(f"  ELO ratings: {match['elo_home']} vs {match['elo_away']}")
            print(f"  League: {match['league']}, Venue: {match['venue']}")
            
            # Basic prediction
            prediction = predictor.predict_single_match(match)
            print("\n🔮 Full-time predictions:")
            print(f"  Final score: {prediction['ft_goals_home']:.1f} - {prediction['ft_goals_away']:.1f}")
            print(f"  Total shots on target: {prediction['ft_sot_home']:.1f} - {prediction['ft_sot_away']:.1f}")
            
            # Prediction with confidence
            if i == 1:  # Only for first match to save time
                print("\n📈 Prediction with uncertainty:")
                confidence = predictor.predict_with_confidence(match, n_samples=50)
                for target, stats in confidence.items():
                    if 'goals' in target:
                        print(f"  {target}: {stats['mean']:.1f} ± {stats['std']:.1f} "
                              f"(95% CI: {stats['ci_lower']:.1f}-{stats['ci_upper']:.1f})")
                
                # Feature importance
                print("\n🎯 Feature importance (for home goals):")
                importance = predictor.get_feature_importance_estimate(match, target_idx=0)
                for feature, imp in list(importance.items())[:5]:  # Top 5
                    print(f"  {feature}: {imp:.4f}")
        
        print(f"\n✅ All predictions completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you've trained the model first by running the main training script.")

if __name__ == "__main__":
    main()
