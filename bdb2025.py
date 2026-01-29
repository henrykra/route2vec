"""Common functions for loading and preparing Big Data Bowl 2025 data for modeling."""
from typing import Optional
from pathlib import Path
import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)

def load_csv_data(
    paths: list[Path]
) -> list[pd.DataFrame]:
    # TODO: look into reading from template, i.e. tracking_week_*.csv
    return [pd.read_csv(path) for path in paths]


def load_tracking_data(
    weeks: Optional[int | list[int]]=None,
    db: Optional[Path]=None,
    folder: Optional[Path]=None,
) -> pd.DataFrame:
   
    MIN_WEEK = 1
    MAX_WEEK = 9

    if not weeks:
        # defualt to all data
        weeks = range(MIN_WEEK, MAX_WEEK)
    
    if isinstance(weeks, int):
        weeks = [weeks]
    weeks = set(weeks)
    
    for week in weeks:
        assert isinstance(week, int) and week >= MIN_WEEK and week <= MAX_WEEK, \
        f"provide integer weeks between {MIN_WEEK} and {MAX_WEEK}"

    logger.info(f"reading data from weeks: {weeks}")

    # TODO: Load from database
    if db and db.exists():
        return

    # load data from a folderectory
    if folder and folder.exists():
        logger.info(f"reading tracking from {folder}")
        paths = [Path(folder, f'tracking_week_{n}.csv') for n in weeks]

        dfs = load_csv_data(paths)
        logger.info("finshed reading tracking")
        return pd.concat(dfs)
    
    raise FileNotFoundError



def load_player_data(
    db: Optional[Path]=None,
    folder: Optional[Path]=None,
) -> pd.DataFrame:
    
    if db and db.exists():
        return

    if folder and folder.exists():
        logger.info(f"reading player data from {folder}")
        path = Path(folder, 'players.csv')

        return pd.read_csv(path)
    

def load_play_data(
        db: Optional[Path]=None,
        folder: Optional[Path]=None,
) -> pd.DataFrame:
    
    if db and db.exists():
        return

    if folder and folder.exists():
        logger.info(f"reading play data from {folder}")
        path = Path(folder, 'plays.csv')

    return pd.read_csv(path)


def clean_player_data(player_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("cleaning player data")
    
    # clean height
    feet = pd.to_numeric(player_df['height'].str.split('-', expand=True)[0])
    inches = pd.to_numeric(player_df['height'].str.split('-', expand=True)[1])
    player_df['height_inches'] = 12 * feet + inches

    # create zscore for height
    player_df['height_z'] = (player_df['height_inches'] - player_df['height_inches'].mean()) / player_df['height_inches'].std()
    player_df['weight_z'] = (player_df['weight'] - player_df['weight'].mean()) / player_df['weight'].std()

    logger.info("finished cleaning player data")
    return player_df


def clean_tracking_data(tracking_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("cleaning tracking data")

    # angles to radians
    tracking_df['o'] = ((-1 * tracking_df['o'] + 90) % 360) * np.pi / 180
    tracking_df['dir'] = ((-1 * tracking_df['dir'] + 90) % 360) * np.pi / 180

    # standardize locations to the ball snap location
    ball = (
        tracking_df
        .loc[
            (tracking_df['event'] == 'ball_snap') & 
            (tracking_df['club'] == 'football'),
             
            ['gameId', 'playId', 'frameId', 'x', 'y']
        ]
    )
    tracking_df = tracking_df.merge(
        ball, # ball info for starting time
        how='left', 
        on=('gameId', 'playId'),
        suffixes=('', '_ball')
    )

    tracking_df['x'] = tracking_df['x'] - tracking_df['x_ball']
    tracking_df['y'] = tracking_df['y'] - tracking_df['y_ball']

    # normalize play directions to the right
    tracking_df['x'] = ((tracking_df['playDirection'] == 'left') * -2 + 1) * tracking_df['x']
    tracking_df['o'] = ( ((tracking_df['playDirection'] == 'left') * np.pi) + tracking_df['o'] ) % (2 * np.pi) 
    tracking_df['dir'] = ( ((tracking_df['playDirection'] == 'left') * np.pi) + tracking_df['dir'] ) % (2 * np.pi) 

    # x and y components of orientation and velocity
    tracking_df['ox'] = np.cos(tracking_df['o'])
    tracking_df['oy'] = np.sin(tracking_df['o'])

    tracking_df['vx'] = np.cos(tracking_df['dir']) * tracking_df['s']
    tracking_df['vy'] = np.sin(tracking_df['dir']) * tracking_df['s']

    tracking_df['stop_point'] = pd.NA
    tracking_df.loc[tracking_df['event'].isin(['pass_outcome_incomplete', 'qb_sack', 'tackle']), 'stop_point'] = True 
    # fill stopping points forward to filter out moments after a defined route stop
    tracking_df['stop_point'] = tracking_df.groupby(['gameId', 'playId', 'nflId'])['stop_point'].ffill()
    
    tracking_df = tracking_df.loc[pd.isna(tracking_df['stop_point'])].copy()

    # filter
    tracking_df = tracking_df.loc[
        (tracking_df['frameId'] >= tracking_df['frameId_ball']) & # after the starting point
        (tracking_df['club'] != 'football')
        
        #['gameId', 'playId', 'frameId', 'nflId', 'x', 'y', 'vx', 'vy', 'a', 'ox', 'oy']
    ].copy()
    
    logger.info("finished cleaning tracking data")
    return tracking_df


def mirror_tracking_plays(tracking_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("mirroring tracking data")
    mirrored_df = tracking_df.copy()

    mirrored_df['y'] = mirrored_df['y'] * -1
    mirrored_df['vy'] = mirrored_df['vy'] * -1
    mirrored_df['oy'] = mirrored_df['oy'] * -1

    mirrored_df['mirrored'] = True
    tracking_df['mirrored'] = False

    logger.info("finished mirroring tracking data")

    return pd.concat([tracking_df, mirrored_df])


def prepare_static_data(tracking_df: pd.DataFrame, play_df: pd.DataFrame, player_df: pd.DataFrame ) -> pd.DataFrame:
    logger.info("preparing static data")
    # join 

    joined_df = (
        tracking_df
        .groupby(['gameId', 'playId', 'nflId', 'mirrored'], as_index=False)
        .first()
        .merge(
            play_df,
            how='left',
            on=('gameId', 'playId')
        )
        .merge( # player info for positions
            player_df, 
            how='left',
            on='nflId'
        )
    )
    joined_df['offense'] = joined_df['club'] == joined_df['possessionTeam']

    # filter
    joined_df = joined_df[        
        ['gameId', 'playId', 'nflId', 'mirrored', 'height_z', 'weight_z', 'position', 'offense']
    ].copy()

    joined_df = pd.get_dummies(joined_df)
    logger.info("finished preparing static data")

    return joined_df

def get_num_static_cols(static_df: pd.DataFrame) -> int:

    return len([col for col in static_df.columns if 'Id' not in col and 'mirrored' not in col])