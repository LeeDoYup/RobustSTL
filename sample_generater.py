import numpy as np
import matplotlib.pyplot as plt

def get_random_choice(total_length, num_choice, replace=False):
    return np.random.choice(total_length, num_choice, replace=replace)

def get_change(change_type, change_level):
    # types: spike, dip, random
    if change_type.lower() in ['random']:
        return np.random.choice([-1,1]) * change_level
    elif change_type.lower() in ['spike', 'increase']:
        return change_level
    elif change_type.lower() in ['dip', 'decrease']:
        return -1.0 * change_level
    else:
        print('[!] NOT VALID change_type: ', change_type)
        raise

def get_season(season_len, season_type, season_level):
    season = np.zeros([season_len])
    if season_type.lower() == 'random':
        season = 2.0*(np.random.random(season_len)-0.5) * season_level
        return season
    elif season_type.lower() == 'stair':
        half_idx = int(season_len/2)
        season[:half_idx] += season_level
        season[half_idx:] -= season_level
        return season
    else:
        print('[!] NOT VALID season type:', season_type)
        raise

def generate_seasons(total_len, season_len, season_num, season_type, season_level):
    season = get_season(season_len, season_type, season_level)
    seasons = np.tile(season, season_num)
    return seasons[:total_len]

def generate_remainders(total_len, noise_mean, noise_std):
    return np.random.normal(noise_mean, noise_std, (total_len,))

def generate_anomalies(total_len, anomaly_num, anomaly_type, anomaly_level):
    anomaly_time_steps = get_random_choice(total_len, anomaly_num)
    anomalies = np.zeros([total_len])
    for item in anomaly_time_steps:
        anomalies[item] += get_change(anomaly_type, anomaly_level)
    return anomalies

def generate_trends(total_len, trend_change_num, trend_type, trend_level):
    trends = np.zeros([total_len])
    change_points = get_random_choice(total_len, trend_change_num)
    for idx, item in enumerate(change_points):
        change_value = get_change(trend_type, trend_level)
        trends[item:] += change_value
    return trends

def sample_generation(total_len=750,
                        season_len=50,
                        season_type='stair',
                        season_level=1,
                        trend_type='random',
                        trend_level=3,
                        trend_change_num=10,
                        anomaly_num=6,
                        anomaly_type='random',
                        anomaly_level=4,
                        noise_mean=0,
                        noise_std=0.316):
    '''
    args:
    - season_type = 'random', 'stair'
    - trend_type = 'random', 'increase', 'decrease'
    - anomaly_type = 'random', 'spike', 'dip'
    '''

    assert total_len >= season_len
    season_num = int(total_len/season_len)+1

    seasons = generate_seasons(total_len, season_len, season_num, season_type, season_level)
    remainders = generate_remainders(total_len, noise_mean, noise_std)
    anomalies = generate_anomalies(total_len, anomaly_num, anomaly_type, anomaly_level)
    trends = generate_trends(total_len, trend_change_num, trend_type, trend_level)
    sample = trends + seasons + remainders + anomalies
    return [sample, trends, seasons, remainders+anomalies]