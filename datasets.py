"""Functions for perparing datasets for modeling"""
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from torch.utils.data import Dataset


def interpolate_movement(tracking_df: pd.DataFrame) -> np.ndarray:

    play_index = tracking_df.set_index(['gameId', 'playId', 'mirrored']).sort_index()

    num_obs = 40
    interpolated_columns = ['x', 'y', 'ox', 'oy', 'vx', 'vy', 'a']
    ids = play_index.index.unique()
    num_plays = len(ids)
    routes_per_play = tracking_df.groupby(['gameId', 'playId'])['nflId'].nunique(dropna=True).max()

    # array for interpolated movement
    movement_arr = np.zeros( (num_plays, routes_per_play, num_obs, len(interpolated_columns)) )

    for i, (gameid, playid, mirrored) in enumerate(ids): 
        
        play: pd.DataFrame = play_index.loc[gameid, playid, mirrored]
        
        for ii, nflid in enumerate(play['nflId'].dropna().unique()):

            player: pd.DataFrame = play.loc[play['nflId'] == nflid]
            min_frame = player['frameId'].min()
            frame_range = player['frameId'].max() - min_frame
            t = np.array((player['frameId'] - min_frame) / frame_range)

            player_data = np.array(player[interpolated_columns])

            # smooth the info about the player over time
            spline = CubicSpline(t, player_data, axis=0, extrapolate=False)

            # take n observations
            observation_times = np.arange(num_obs) / num_obs

            # populate an array with smoothed values
            movement_arr[i, ii, ...] = spline(observation_times)

    return movement_arr


def perpare_data_arrays(static_df: pd.DataFrame, movement_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    play_index = static_df.set_index(['gameId', 'playId', 'mirrored'])
    
    num_plays = len(static_df.index)
    routes_per_play = static_df.groupby(['gameId', 'playId', 'mirrored'])['nflId'].nunique(dropna=True).iloc[0]

    assert (static_df.groupby(['gameId', 'playId', 'mirrored'])['nflId'].nunique(dropna=False) == routes_per_play).all(), \
    "Different values across player dimension"
    
    num_columns = len(play_index.columns) - 1 # subtract the nflId column
    player_info = np.zeros( (num_plays, routes_per_play, num_columns) ) 

    for i, (name, group) in enumerate(play_index.groupby(['gameId', 'playId', 'mirrored'])):
        player_info[i, ...] = group.drop(columns=['nflId'])

    input_arr = np.concat(
        [
            movement_arr.reshape((num_plays, routes_per_play, -1)), # combine final axes 
            player_info, # concat fixed info
        ],
        axis=-1
    )

    target_arr = movement_arr[..., :2, :].reshape((num_plays, routes_per_play, -1))

    return input_arr, target_arr


def mask_input(input_arr: np.ndarray, n: int) -> np.ndarray:
    mask_pct = .15

    # "mask token"
    mask = np.zeros(n)

    n_player_vectors = np.prod(input_arr.shape[:-1])
    mask_idx = np.random.choice(
        n_player_vectors,
        size=int(n_player_vectors * mask_pct),
        replace=False
    ) # random choice over the product of the first axes

    # determine vectors to mask

    masked_players_arr = input_arr.reshape((n_player_vectors, -1)).copy()
    masked_players_arr[mask_idx, :n] = mask

    masked_players_arr.reshape(input_arr.shape)

    return masked_players_arr


class NflDataset(Dataset):
    
    def __init__(self, input_arr: np.ndarray, target_arr: np.ndarray):

        # make sure there's the right amount of input observatinos and target observations
        assert input_arr.shape[0] == target_arr.shape[0], "# Observations mismatch between input and target"

        self.input_arr = input_arr
        self.target_arr = target_arr

        return
    
    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray]:
        
        return self.input_arr[idx], self.target_arr[idx]


    def __len__(self) -> int:

        return self.input_arr.shape[0]
    
