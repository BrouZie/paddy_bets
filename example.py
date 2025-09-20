from football_predictor import FootballMatchPredictor

# Load the trained model
predictor = FootballMatchPredictor()

# Define a match (half-time situation)
match = {
    "ht_goals_home": 2,
    "ht_goals_away": 0,
    "ht_sot_home": 4,
    "ht_sot_away": 0,
    "ht_reds_home": 0,
    "ht_reds_away": 1,
    "elo_home": 1699,
    "elo_away": 1993,
    "league": "Premier League",
    "season": "2023-24",
    "venue": "Home"
}

# Get prediction
prediction = predictor.predict_single_match(match)
print(f"Predicted final score: {prediction['ft_goals_home']:.1f} - {prediction['ft_goals_away']:.1f}")

# Get uncertainty estimate
confidence = predictor.predict_with_confidence(match)
print(f"Home goals 95% CI: {confidence['ft_goals_home']['ci_lower']:.1f} - {confidence['ft_goals_home']['ci_upper']:.1f}")
