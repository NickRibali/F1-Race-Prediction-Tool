"""Model code for the F1 race predictor: features, training, and prediction."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ['grid_position', 'driver_form', 'team_form', 'driver_skill_component',
            'driver_disruption_rate', 'team_disruption_rate',
            'regulation_era_pre-2022', 'cost_cap_era_pre-2021']
NEEDED = ['race_key', 'driver_abbr', 'team_name', 'grid_position', 'finish_position',
          'regulation_era', 'cost_cap_era', 'disruption_flag']


def prior_mean(data, group, value, window, new_col):
    """Average of the previous `window` races. shift(1) keeps the current race out."""
    per_race = (data.groupby([group, 'race_key'], as_index=False)[value].mean()
                    .sort_values([group, 'race_key']))
    per_race[new_col] = (per_race.groupby(group)[value]
                         .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean()))
    return data.merge(per_race[[group, 'race_key', new_col]], on=[group, 'race_key'], how='left')


def build_features(clean_df):
    missing = [c for c in NEEDED if c not in clean_df.columns]
    if missing:
        raise ValueError(f"f1_clean.csv is missing columns: {missing}")
    df = clean_df.sort_values(['race_key', 'finish_position']).reset_index(drop=True)
    df = prior_mean(df, 'driver_abbr', 'finish_position', 3, 'driver_form')
    df = prior_mean(df, 'team_name', 'finish_position', 5, 'team_form')
    df = prior_mean(df, 'driver_abbr', 'disruption_flag', 10, 'driver_disruption_rate')
    df = prior_mean(df, 'team_name', 'disruption_flag', 10, 'team_disruption_rate')
    df['driver_skill_component'] = df['driver_form'] - df['team_form']
    df['regulation_era_pre-2022'] = (df['regulation_era'] == 'pre-2022').astype(int)
    df['cost_cap_era_pre-2021'] = (df['cost_cap_era'] == 'pre-2021').astype(int)
    return df.sort_values('race_key', kind='stable').reset_index(drop=True)


def new_model(alpha=0.1):
    return Pipeline([('impute', SimpleImputer(strategy='mean')),
                     ('scale', StandardScaler()),
                     ('model', Ridge(alpha=alpha))])


def latest_form(history, group, value, window):
    per_race = (history.groupby([group, 'race_key'], as_index=False)[value].mean()
                       .sort_values([group, 'race_key']))
    return per_race.groupby(group)[value].apply(lambda s: s.tail(window).mean())


class Predictor:
    """Trains on every race in the data and predicts the order for a new grid."""

    def __init__(self, clean_df):
        feat = build_features(clean_df)
        self.model = new_model().fit(feat[FEATURES], feat['finish_position'])
        self.n_rows = len(feat)
        self.n_races = feat['race_key'].nunique()
        self.latest_race_key = int(feat['race_key'].max())
        self.known_drivers = set(feat['driver_abbr'])
        self.known_teams = set(feat['team_name'])
        self.tables = {
            'driver_form': latest_form(feat, 'driver_abbr', 'finish_position', 3),
            'team_form': latest_form(feat, 'team_name', 'finish_position', 5),
            'driver_dnf': latest_form(feat, 'driver_abbr', 'disruption_flag', 10),
            'team_dnf': latest_form(feat, 'team_name', 'disruption_flag', 10)}

    def check_grid(self, grid_df):
        """Returns (clean grid or None, errors, warnings)."""
        g = grid_df.dropna(how='all').copy()
        if g.empty:
            return None, ["Add at least one driver."], []
        if len(g) > 24:
            return None, ["A grid can have at most 24 drivers."], []
        if g[['driver_abbr', 'team_name', 'grid_position']].isna().any().any():
            return None, ["Every row needs a driver, a team and a grid position."], []
        g['driver_abbr'] = g['driver_abbr'].astype(str).str.strip().str.upper()
        g['team_name'] = g['team_name'].astype(str).str.strip()
        g['grid_position'] = pd.to_numeric(g['grid_position'], errors='coerce')
        if g['grid_position'].isna().any() or not g['grid_position'].between(1, 30).all():
            return None, ["Grid positions must be numbers from 1 to 30."], []
        if g['driver_abbr'].duplicated().any():
            return None, ["A driver appears more than once."], []
        warnings = [f"No history for driver {d}. Averages were used."
                    for d in sorted(set(g['driver_abbr']) - self.known_drivers)]
        warnings += [f"No history for team {t}. Averages were used."
                     for t in sorted(set(g['team_name']) - self.known_teams)]
        return g.reset_index(drop=True), [], warnings

    def predict(self, grid):
        t = self.tables
        X = pd.DataFrame(index=grid.index, columns=FEATURES, dtype=float)
        X['grid_position'] = grid['grid_position']
        X['driver_form'] = grid['driver_abbr'].map(t['driver_form'])
        X['team_form'] = grid['team_name'].map(t['team_form'])
        X['driver_skill_component'] = X['driver_form'] - X['team_form']
        X['driver_disruption_rate'] = grid['driver_abbr'].map(t['driver_dnf'])
        X['team_disruption_rate'] = grid['team_name'].map(t['team_dnf'])
        X['regulation_era_pre-2022'] = 0     # the next race is in the current rules era
        X['cost_cap_era_pre-2021'] = 0
        out = grid.copy()
        score = pd.Series(self.model.predict(X[FEATURES]), index=out.index)
        out['predicted_finish'] = score.rank(method='first').astype(int)
        out['recent_dnf_rate'] = grid['driver_abbr'].map(t['driver_dnf']).round(2)
        return out.sort_values('predicted_finish').reset_index(drop=True)


def walk_forward(clean_df, last_k=5):
    """Train on earlier races, predict the next one, for each of the last k races."""
    feat = build_features(clean_df)
    actual, pred, grid = [], [], []
    for rk in sorted(feat['race_key'].unique())[-last_k:]:
        train, test = feat[feat['race_key'] < rk], feat[feat['race_key'] == rk]
        model = new_model().fit(train[FEATURES], train['finish_position'])
        actual += list(test['finish_position'])
        pred += list(model.predict(test[FEATURES]))
        grid += list(test['grid_position'])
    actual, pred, grid = map(np.array, (actual, pred, grid))
    return pd.DataFrame({
        'Model': [np.abs(actual - pred).mean(), np.sqrt(((actual - pred) ** 2).mean())],
        'Finish = grid baseline': [np.abs(actual - grid).mean(), np.sqrt(((actual - grid) ** 2).mean())],
    }, index=['Average miss (MAE)', 'Big-miss score (RMSE)']).round(3)
