#!/usr/bin/env python3
"""
Interactive Football Match Predictor CLI
Run this script to input match data and get predictions interactively
"""

import sys
from football_predictor import FootballMatchPredictor
from typing import Dict, Union

def get_user_input() -> Dict[str, Union[int, float, str]]:
    """Get match data from user input"""
    print("\n🏈 Enter Half-Time Match Details")
    print("=" * 40)
    
    match_data = {}
    
    # Numeric inputs
    numeric_fields = [
        ("ht_goals_home", "Home team goals at half-time", int),
        ("ht_goals_away", "Away team goals at half-time", int),
        ("ht_sot_home", "Home team shots on target at half-time", int),
        ("ht_sot_away", "Away team shots on target at half-time", int),
        ("ht_reds_home", "Home team red cards at half-time", int),
        ("ht_reds_away", "Away team red cards at half-time", int),
        ("elo_home", "Home team ELO rating", int),
        ("elo_away", "Away team ELO rating", int),
    ]
    
    for field, description, data_type in numeric_fields:
        while True:
            try:
                value = input(f"{description}: ")
                match_data[field] = data_type(value)
                break
            except ValueError:
                print(f"Please enter a valid number for {description}")
    
    # Categorical inputs with validation
    print("\nLeague options: Premier League, La Liga, Bundesliga, Serie A, Ligue 1, or custom")
    match_data["league"] = input("League: ").strip() or "Premier League"
    
    print("Season options: 2020-21, 2021-22, 2022-23, 2023-24")
    match_data["season"] = input("Season: ").strip() or "2023-24"
    
    print("Venue options: Home, Away, Neutral")
    while True:
        venue = input("Venue: ").strip() or "Home"
        if venue in ["Home", "Away", "Neutral"]:
            match_data["venue"] = venue
            break
        print("Please enter 'Home', 'Away', or 'Neutral'")
    
    return match_data

def display_prediction(match_data: Dict, prediction: Dict, predictor: FootballMatchPredictor):
    """Display prediction results in a nice format"""
    print("\n" + "="*50)
    print("🔮 FULL-TIME PREDICTION RESULTS")
    print("="*50)
    
    # Current situation
    print(f"\n📊 Half-Time Situation:")
    print(f"  Score: {match_data['ht_goals_home']}-{match_data['ht_goals_away']}")
    print(f"  Shots on Target: {match_data['ht_sot_home']}-{match_data['ht_sot_away']}")
    print(f"  Red Cards: {match_data['ht_reds_home']}-{match_data['ht_reds_away']}")
    print(f"  ELO Ratings: {match_data['elo_home']} vs {match_data['elo_away']}")
    print(f"  Match: {match_data['league']}, {match_data['season']}, {match_data['venue']} venue")
    
    # Predictions
    print(f"\n🎯 Full-Time Predictions:")
    home_goals = prediction['ft_goals_home']
    away_goals = prediction['ft_goals_away']
    home_sot = prediction['ft_sot_home']
    away_sot = prediction['ft_sot_away']
    
    print(f"  Final Score: {home_goals:.1f} - {away_goals:.1f}")
    print(f"  Total Shots on Target: {home_sot:.1f} - {away_sot:.1f}")
    
    # Additional analysis
    print(f"\n📈 Analysis:")
    goals_diff = home_goals - away_goals
    if goals_diff > 0.5:
        print(f"  • Home team likely to win by {goals_diff:.1f} goals")
    elif goals_diff < -0.5:
        print(f"  • Away team likely to win by {abs(goals_diff):.1f} goals")
    else:
        print(f"  • Very close match expected")
    
    total_goals = home_goals + away_goals
    if total_goals > 3.5:
        print(f"  • High-scoring game expected ({total_goals:.1f} total goals)")
    elif total_goals < 2:
        print(f"  • Low-scoring game expected ({total_goals:.1f} total goals)")
    else:
        print(f"  • Moderate scoring expected ({total_goals:.1f} total goals)")
    
    # Second half predictions
    h2_home_goals = max(0, home_goals - match_data['ht_goals_home'])
    h2_away_goals = max(0, away_goals - match_data['ht_goals_away'])
    print(f"  • Expected 2nd half goals: {h2_home_goals:.1f} - {h2_away_goals:.1f}")

def quick_examples():
    """Show some quick example predictions"""
    print("\n🚀 Quick Example Predictions")
    print("="*40)
    
    examples = [
        {
            "name": "Manchester City vs Arsenal (close at HT)",
            "data": {
                "ht_goals_home": 1, "ht_goals_away": 1,
                "ht_sot_home": 4, "ht_sot_away": 3,
                "ht_reds_home": 0, "ht_reds_away": 0,
                "elo_home": 1750, "elo_away": 1680,
                "league": "Premier League", "season": "2023-24", "venue": "Home"
            }
        },
        {
            "name": "Real Madrid vs Getafe (dominating)",
            "data": {
                "ht_goals_home": 2, "ht_goals_away": 0,
                "ht_sot_home": 6, "ht_sot_away": 1,
                "ht_reds_home": 0, "ht_reds_away": 0,
                "elo_home": 1800, "elo_away": 1450,
                "league": "La Liga", "season": "2023-24", "venue": "Home"
            }
        }
    ]
    
    try:
        predictor = FootballMatchPredictor()
        for example in examples:
            print(f"\n📌 {example['name']}")
            prediction = predictor.predict_single_match(example['data'])
            print(f"   HT: {example['data']['ht_goals_home']}-{example['data']['ht_goals_away']} → "
                  f"FT: {prediction['ft_goals_home']:.1f}-{prediction['ft_goals_away']:.1f}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

def main():
    """Main interactive loop"""
    print("🏈 Interactive Football Match Predictor")
    print("=" * 50)
    
    # Try to load the model
    try:
        predictor = FootballMatchPredictor()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nPlease make sure you have:")
        print("1. Trained the model (run main.py)")
        print("2. The files torch_ht2ft.pt and preprocessor.pkl exist")
        return
    
    while True:
        print("\n" + "="*50)
        print("Choose an option:")
        print("1. Enter custom match data")
        print("2. See quick examples")
        print("3. Exit")
        print("="*50)
        
        choice = input("Your choice (1-3): ").strip()
        
        if choice == "1":
            try:
                # Get user input
                match_data = get_user_input()
                
                # Make prediction
                print("\n⏳ Making prediction...")
                prediction = predictor.predict_single_match(match_data)
                
                # Display results
                display_prediction(match_data, prediction, predictor)
                
                # Ask for confidence intervals
                conf_choice = input("\nShow prediction uncertainty? (y/n): ").lower()
                if conf_choice in ['y', 'yes']:
                    print("⏳ Calculating uncertainty (this may take a moment)...")
                    confidence = predictor.predict_with_confidence(match_data, n_samples=30)
                    
                    print("\n📊 Prediction Ranges (95% confidence):")
                    for target, stats in confidence.items():
                        if 'goals' in target:
                            team = "Home" if 'home' in target else "Away"
                            print(f"  {team} goals: {stats['ci_lower']:.1f} - {stats['ci_upper']:.1f} "
                                  f"(most likely: {stats['mean']:.1f})")
                
            except KeyboardInterrupt:
                print("\n\nOperation cancelled.")
            except Exception as e:
                print(f"\n❌ Error making prediction: {e}")
        
        elif choice == "2":
            quick_examples()
        
        elif choice == "3":
            print("\n👋 Thanks for using the Football Match Predictor!")
            break
        
        else:
            print("Invalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
