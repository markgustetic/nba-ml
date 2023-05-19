import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

nba_file_path = "./data/nba_game_data.csv"
nba_data = pd.read_csv(nba_file_path)

y = nba_data["HOME_TEAM_WINS"]

nba_features = ["PTS_home", "PTS_away"]

X = nba_data[nba_features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

nba_model = RandomForestRegressor(random_state=1)
nba_model.fit(train_X, train_y)

nba_preds = nba_model.predict(val_X)

print(mean_absolute_error(val_y, nba_preds))
