"""
Generates a synthetic football match dataset so the pipeline runs end-to-end
without needing API keys first. Replace this with real data via
src/data_ingestion.py once you're ready (see README).

Features per match (from the HOME team's perspective):
  - home_elo, away_elo            -> Elo-style strength rating
  - home_fifa_rating, away_fifa_rating -> proxy for FIFA/game ratings (0-99)
  - home_form_pts, away_form_pts  -> points won in last 5 matches (0-15)
  - home_goals_avg, away_goals_avg -> average goals scored, last 10 matches
  - h2h_home_win_rate              -> historic head-to-head win rate for home team
  - home_advantage                 -> constant 1 (home team always has this)

Target:
  - result: 'H' (home win), 'D' (draw), 'A' (away win)
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
TEAMS = [
    "Arsenal", "Man City", "Liverpool", "Chelsea", "Tottenham",
    "Man United", "Newcastle", "Aston Villa", "Brighton", "West Ham",
    "Everton", "Wolves", "Fulham", "Brentford", "Crystal Palace",
    "Bournemouth", "Nottingham Forest", "Leicester", "Southampton", "Ipswich",
]

def make_team_ratings(teams):
    """Assign each team a latent 'true strength' so outcomes correlate with features."""
    return {t: RNG.normal(loc=1500, scale=120) for t in teams}

def simulate_dataset(n_matches=3000):
    strengths = make_team_ratings(TEAMS)
    rows = []
    for _ in range(n_matches):
        home, away = RNG.choice(TEAMS, size=2, replace=False)
        home_elo = strengths[home] + RNG.normal(0, 40)
        away_elo = strengths[away] + RNG.normal(0, 40)

        home_fifa = np.clip((home_elo - 1300) / 6 + RNG.normal(0, 4), 50, 95)
        away_fifa = np.clip((away_elo - 1300) / 6 + RNG.normal(0, 4), 50, 95)

        home_form = np.clip(RNG.normal((home_elo - 1400) / 40, 3), 0, 15)
        away_form = np.clip(RNG.normal((away_elo - 1400) / 40, 3), 0, 15)

        home_goals_avg = np.clip(RNG.normal(1.4 + (home_elo - 1500) / 400, 0.4), 0.2, 3.5)
        away_goals_avg = np.clip(RNG.normal(1.2 + (away_elo - 1500) / 400, 0.4), 0.2, 3.5)

        h2h_home_win_rate = np.clip(
            0.5 + (home_elo - away_elo) / 800 + RNG.normal(0, 0.08), 0.05, 0.95
        )

        # Whether this match is at the home team's actual home stadium (1)
        # or a neutral venue like a cup final (0). ~90% are true home games.
        true_home_venue = 1 if RNG.random() < 0.9 else 0

        # Outcome probability driven by elo diff + home advantage (logistic-ish)
        # Home advantage only applies at the true home venue -- e.g. Barcelona
        # at Camp Nou gets the boost; Barcelona at a neutral final doesn't.
        home_advantage_boost = 80 if true_home_venue else 0
        elo_diff = (home_elo + home_advantage_boost) - away_elo
        p_home = 1 / (1 + np.exp(-elo_diff / 200))
        p_draw = 0.24 + RNG.normal(0, 0.02)
        p_draw = np.clip(p_draw, 0.15, 0.32)
        p_home = p_home * (1 - p_draw)
        p_away = 1 - p_home - p_draw

        result = RNG.choice(["H", "D", "A"], p=[p_home, p_draw, p_away])

        rows.append(dict(
            home_team=home, away_team=away,
            home_elo=round(home_elo, 1), away_elo=round(away_elo, 1),
            home_fifa_rating=round(home_fifa, 1), away_fifa_rating=round(away_fifa, 1),
            home_form_pts=round(home_form, 1), away_form_pts=round(away_form, 1),
            home_goals_avg=round(home_goals_avg, 2), away_goals_avg=round(away_goals_avg, 2),
            h2h_home_win_rate=round(h2h_home_win_rate, 3),
            true_home_venue=true_home_venue,
            result=result,
        ))

    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = simulate_dataset(3000)
    out_path = "data/matches.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(df["result"].value_counts(normalize=True))
