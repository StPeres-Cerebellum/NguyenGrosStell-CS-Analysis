from pywavesurfer import ws
# from pywavesurfer import loadDataFile
import pandas as pd
import numpy as np
import os
import re
from scipy.interpolate import interp1d
from scipy import stats
from pathlib import Path
from ScanImageTiffReader import ScanImageTiffReader
from datetime import datetime

def unpackbits(x, num_bits):
    if np.issubdtype(x.dtype, np.floating):
        raise ValueError("numpy data type needs to be int-like")
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2 ** np.arange(num_bits, dtype=x.dtype).reshape([1, num_bits])
    return (x & mask).astype(bool).astype(int).reshape(xshape + [num_bits])


def fast_load_wavesurfer_binary_into_pandas(h5_file_path, only_changes=True):
    wave_surfer_dict = ws.loadDataFile(filename=h5_file_path, format_string='raw')
    channel_names = [chan.decode('utf-8') for chan in wave_surfer_dict['header']['DIChannelNames']]
    sweeps = sorted([sweep for sweep in wave_surfer_dict if sweep.startswith('sweep_')])
    arrays = [wave_surfer_dict[sweep]['digitalScans'][0] for sweep in wave_surfer_dict if sweep.startswith('sweep_')]
    try:
        df = pd.DataFrame(np.vstack(arrays).flatten(), columns=['BINARY'])
    except ValueError:
        df = pd.DataFrame(np.vstack(arrays[:-1]).flatten(), columns=['BINARY'])
        sweeps = sweeps[:-1]
    if only_changes:
        df = df.loc[df['BINARY'].diff() != 0]
    df['sweep'] = (df.index // wave_surfer_dict['header']['ExpectedSweepScanCount'][0][0]).astype('int')
    df['sweep_name'] = df['sweep'] + int(sweeps[0].replace('sweep_', ''))
    df['sweep'] += 1
    df['sweep_num'] = df['sweep']
    df['sweep_time'] = df.index / wave_surfer_dict['header']['AcquisitionSampleRate'][0][0]
    df['total_time'] = df['sweep_time']
    df[list(channel_names)] = unpackbits(df['BINARY'].values.reshape((-1, 1)), len(channel_names)).reshape(-1,
                                                                                                           len(channel_names))
    return df.reset_index(names=['sample_number'])


def raw_encoder_to_steps(encoder_a, encoder_b):
    df_encoder = pd.DataFrame(data=encoder_a.values, columns=['a'], index=encoder_a.index)
    df_encoder['b'] = encoder_b.values
    df_encoder['a_change'] = df_encoder.a.diff().replace({0: np.nan})
    df_encoder['b_change'] = df_encoder.b.diff().replace({0: np.nan})
    # calculate steps forward
    df_encoder['steps'] = 0
    df_encoder.steps += ((df_encoder.a_change + df_encoder.b) == 1).astype(int)
    df_encoder.steps += ((df_encoder.a_change + df_encoder.b) == 0).astype(int)
    df_encoder.steps += ((df_encoder.b_change + df_encoder.a) == 2).astype(int)
    df_encoder.steps += ((df_encoder.b_change + df_encoder.a) == -1).astype(int)
    # include steps backward
    df_encoder.steps -= ((df_encoder.a_change + df_encoder.b) == 2).astype(int)
    df_encoder.steps -= ((df_encoder.a_change + df_encoder.b) == -1).astype(int)
    df_encoder.steps -= ((df_encoder.b_change + df_encoder.a) == 1).astype(int)
    df_encoder.steps -= ((df_encoder.b_change + df_encoder.a) == 0).astype(int)
    return df_encoder.steps


def mark_false_starts_and_move_tests(encoder_df):
    encoder_df[['motor_zero_count', 'joystick_zero_count']] = encoder_df.groupby('sweep').transform('sum')[
        ['motor_zero', 'joystick_zero']].copy()
    encoder_df['false_start'] = 0
    encoder_df.loc[(encoder_df['motor_zero_count'] == 0) & (encoder_df['joystick_zero_count'] == 0), 'false_start'] = 1
    known_false_starts = encoder_df.loc[
        (encoder_df.joystick_zero_count == 0) & (encoder_df.motor_zero_count > 0) & (encoder_df.solenoid == 1)][
        'sweep'].unique()
    potential_move_tests = encoder_df.loc[(encoder_df.joystick_zero_count == 0) & (encoder_df.motor_zero_count > 0)][
        'sweep'].unique()
    encoder_df.loc[encoder_df.sweep.isin(known_false_starts), 'false_start'] = 1
    encoder_df['move_test'] = np.nan
    encoder_df['stop_test'] = np.nan
    threshold = encoder_df.motorPos.quantile(0.5)
    potential_move_tests = encoder_df.groupby('sweep')['solenoid'].max().reset_index()
    potential_move_tests = potential_move_tests.loc[potential_move_tests.solenoid == 0]['sweep'].values
    move_test_sweeps = \
    encoder_df.loc[(encoder_df.sweep.isin(potential_move_tests)) & (encoder_df.motorPos > threshold)]['sweep'].unique()
    encoder_df.loc[encoder_df.sweep.isin(move_test_sweeps), 'move_test'] = 2
    encoder_df.loc[encoder_df.sweep.isin(move_test_sweeps), 'gain_on'] = 2
    encoder_df.loc[encoder_df['move_test'] == 2, ['pull_number_since_gain', 'false_start']] = [1, 0]
    encoder_df.loc[encoder_df['stop_test'] == 1, 'false_start'] = 0
    encoder_df.loc[(encoder_df.gain_on == 2), 'pull_number_since_gain'] = 1
    if len(move_test_sweeps) < 10:  # if there were fewer than 10 then they must be false starts
        encoder_df.loc[encoder_df['move_test'] == 1, 'false_start'] = 1
    return encoder_df

def mark_rewarded_sweeps(encoder_df):
    rewarded_sweeps = encoder_df.loc[encoder_df['reward_pumping'] == 1, 'sweep'].unique() + 1 # the reward is delivered at the end of the previous sweep when the solenoid is down
    encoder_df['Rewarded'] = 0
    encoder_df.loc[encoder_df['sweep'].isin(rewarded_sweeps), 'Rewarded'] = 1
    encoder_df['Reward'] = 'Omitted'
    encoder_df.loc[encoder_df['Rewarded'] == 1, 'Reward'] = 'Delivered'
    return encoder_df

def zero_sweeped_encoder_sweeps(df, is_continuous=False):
    if is_continuous:
        df['xPos'] = raw_encoder_to_steps(df['xPos_a'], df['xPos_b']).cumsum()
        df['motorPos'] = raw_encoder_to_steps(df['motorPos_a'], df['motorPos_b']).cumsum()
        return df
    for sweep in df.sweep.unique():
        xPos_a = df.loc[df.sweep == sweep, 'xPos_a']
        xPos_b = df.loc[df.sweep == sweep, 'xPos_b']
        df.loc[df.sweep == sweep, 'xPos'] = raw_encoder_to_steps(xPos_a, xPos_b).cumsum()

        motorPos_a = df.loc[df.sweep == sweep, 'motorPos_a']
        motorPos_b = df.loc[df.sweep == sweep, 'motorPos_b']
        df.loc[df.sweep == sweep, 'motorPos'] = raw_encoder_to_steps(motorPos_a, motorPos_b).cumsum()
    return df


def mark_trigger_starts(encoder_df):
    if 'trigger_start' not in encoder_df.columns:
        encoder_df['trigger_start'] = 0
        encoder_df.loc[encoder_df['trigger'].diff() > 0, 'trigger_start'] = 1
    return encoder_df

def is_continuous_recording_with_sweeps(encoder_df):
    is_continuous = False
    if 'trigger' in encoder_df.columns:
        encoder_df = mark_trigger_starts(encoder_df)
        if 'original_sweep' in encoder_df.columns:
            is_continuous = True if encoder_df.groupby('original_sweep')['trigger_start'].sum().median() > 3 else False
        else:
            is_continuous = True if encoder_df.groupby('sweep')['trigger_start'].sum().median() > 3 else False
    return encoder_df, is_continuous

def change_sweep_times_for_continuous_recordings_with_triggers_behavior(encoder_df, pre_pull_buffer=1.):
    trigger_times = encoder_df.loc[encoder_df['trigger_start'] == 1, 'total_time'].values
    if len(trigger_times) > 0:
        if 'original_sweep_time' in encoder_df.columns:
            encoder_df.drop(columns=['original_sweep_time'], inplace=True)
        encoder_df.rename(columns={'sweep_time': 'original_sweep_time'}, inplace=True)
        encoder_df['sweep_start_time'] = np.nan
        encoder_df['sweep_start'] = 0
    for trigger_time in trigger_times:
        encoder_df.loc[(encoder_df['total_time'] < trigger_time - pre_pull_buffer) & (
                    encoder_df['total_time'].shift(-1) >= trigger_time - pre_pull_buffer), ['sweep_start_time', 'sweep_start']] = [trigger_time, 1]
    encoder_df['sweep_start_time'] = encoder_df['sweep_start_time'].ffill()
    encoder_df['sweep_start_time'] = encoder_df['sweep_start_time'].bfill()
    encoder_df['sweep_time'] = encoder_df['total_time'] - encoder_df['sweep_start_time']
    return encoder_df

def change_sweep_numbers_for_continuous_recordings_with_triggers_behavior(encoder_df):
    if 'original_sweep' in encoder_df.columns:
        encoder_df.drop(columns=['original_sweep'], inplace=True)
    encoder_df.rename(columns={'sweep': 'original_sweep'}, inplace=True)
    encoder_df['sweep_start'] = 0
    encoder_df.loc[encoder_df['sweep_time'].diff() < 0, 'sweep_start'] = 1
    encoder_df.loc[0, 'sweep_start'] = 1
    encoder_df['sweep'] = encoder_df['sweep_start'].cumsum()
    encoder_df['sweep_num'] = encoder_df['sweep']
    return encoder_df


def change_gains_for_continuous_recordings_with_triggers_behavior(encoder_df):
    if 'original_gain_on' in encoder_df.columns:
        encoder_df.drop(columns=['original_gain_on'], inplace=True)
    encoder_df.rename(columns={'gain_on': 'original_gain_on'}, inplace=True)
    encoder_df['gain_on'] = np.nan
    gains = encoder_df.loc[encoder_df['trigger_start'] == 1, ['sweep', 'original_gain_on']]
    for index, row in gains.iterrows():
        encoder_df.loc[encoder_df['sweep'] == row['sweep'], 'gain_on'] = row['original_gain_on']
    return encoder_df

def scale_encoder_data(df, pulley_diameter=(0.051 * 2.8), joystick_length=0.13, joystick_ticks=4096, pulley_ticks=2048,
                       lin_motor_scale_factor=10e-6, acq4=False, v_median_filter=50, round_sweep_time_to=0.001):
    joystick_scale_factor = ((joystick_length * 2 * np.pi) / (joystick_ticks * 4))
    motor_scale_factor = ((pulley_diameter * np.pi) / (pulley_ticks * 4))
    if not acq4:
        df['xPos'] = df['xPos'].cumsum()
        df['motorPos'] = df['motorPos'].cumsum()
    df['xPos'] *= joystick_scale_factor
    df['motorPos'] *= motor_scale_factor
    if['linMotorPos'] in df.columns.values:
        df['linMotorPos'] = df['linMotorPos'].cumsum()
        df['linMotorPos'] *= lin_motor_scale_factor
    df[['dx', 'dM', 'dt']] = df.loc[df['xPos'].diff() != 0, ['xPos', 'motorPos', 'total_time']]
    df['xV'] = df['dx'].dropna().diff() / df['dt'].dropna().diff()
    df['xV'] = df['xV'].ffill()
    df['xV_smooth'] = df['xV'].dropna().rolling(v_median_filter).median().shift(-int(v_median_filter/2))

    df['mV'] = df['dM'].dropna().diff() / df['dt'].dropna().diff()
    df['mV'] = df['mV'].ffill()
    df['mV_smooth'] = df['mV'].dropna().rolling(v_median_filter).median().shift(-int(v_median_filter/2))
    return df

def mark_zero_crossings_per_sweep(encoder_df):
    encoder_df[['motor_zero_count', 'joystick_zero_count']] = encoder_df.groupby('sweep').transform('sum')[
        ['motor_zero', 'joystick_zero']].copy()
    return encoder_df

def mark_false_starts_and_move_tests2(df, thresh=0.3):
    df['false_start'] = 0
    df['move_test'] = 0
    xPos_thresh = df.loc[df['gain_on'] == 1, 'xPos'].quantile(thresh)
    motorPos_thresh = df.loc[df['gain_on'] == 1, 'motorPos'].quantile(thresh)
    df['motor_false_start'] = (df.groupby('sweep')['motorPos'].transform(lambda x: np.percentile(x, 95)) < motorPos_thresh).astype(int)
    df['false_start'] = (df.groupby('sweep')['xPos'].transform(lambda x: np.percentile(x, 95)) < xPos_thresh).astype(int)
    # df.loc[(df['motor_false_start'] == 1) & (df['xPos_false_start'] == 1), 'false_start'] = 1
    for sweep in df['sweep'].unique():
        if ((df.loc[df['sweep'] == sweep, 'motor_false_start'] == 0).all() and
                (df.loc[df['sweep'] == sweep, 'solenoid'] == 0).all()):
            df.loc[df['sweep'] == sweep, 'move_test'] = 2
    df['catch_trial'] = 0
    for sweep in df['sweep'].unique():
        if ((df.loc[df['sweep'] == sweep, 'motor_false_start'] == 1).all() and
                (df.loc[df['sweep'] == sweep, 'solenoid'] == 1).any()):
            df.loc[df['sweep'] == sweep, 'catch_trial'] = 1
            df.loc[df['sweep'] == sweep, 'false_start'] = 1
    df.loc[df['move_test'] == 2, 'false_start'] = 0
    return df

def mark_sweep_number_since_gain_change(encoder_df, exclude_false_starts=True):
    encoder_df['gain_on'] = encoder_df['gain_on'].ffill()
    encoder_df['gain_count'] = 0
    encoder_df.loc[encoder_df.gain_on.diff() != 0, 'gain_count'] = 1
    encoder_df['gain_count'] = encoder_df['gain_count'].cumsum()
    # force each sweep to have only one gain count:
    encoder_df.groupby('sweep')['gain_count'].transform('median')
    encoder_df.iloc[0, encoder_df.columns.get_loc('sweep_start')] = 1
    if 'move_test' not in encoder_df.columns:
        encoder_df['move_test'] = 0
    if 'false_start' not in encoder_df.columns:
        encoder_df['false_start'] = 0
    if 'catch_trial' not in encoder_df.columns:
        encoder_df['catch_trial'] = 0
    if exclude_false_starts:
        # initialize column with 0s
        encoder_df['pull_number_since_gain'] = 0

        # compute cumsum for non-false starts
        valid_rows = encoder_df['false_start'] == 0
        encoder_df.loc[valid_rows, 'pull_number_since_gain'] = (
            encoder_df.loc[valid_rows]
            .groupby('gain_count')['sweep_start']
            .cumsum()
        )
    else:
        encoder_df['pull_number_since_gain'] = encoder_df.groupby('gain_count')['sweep_start'].cumsum()
    encoder_df.loc[encoder_df['move_test'] == 2, 'pull_number_since_gain'] = 1
    encoder_df.loc[encoder_df['false_start'] == 1, 'pull_number_since_gain'] = 0
    encoder_df.loc[encoder_df['catch_trial'] == 1, 'pull_number_since_gain'] = 1
    return encoder_df

def mark_control_pulls(encoder_df, pull_number_label=1):
    gain_counter = encoder_df['gain_on'].diff().abs().cumsum()
    gain_counter.iloc[0] = 0
    encoder_df.loc[gain_counter < 1, 'pull_number_since_gain'] = pull_number_label
    encoder_df.loc[gain_counter < 1, 'gain_on'] = -1
    encoder_df.loc[encoder_df['false_start'] == 1, 'pull_number_since_gain'] = 0
    if 'catch_trial' in encoder_df.columns:
        encoder_df.loc[encoder_df['catch_trial'] == 1, 'pull_number_since_gain'] = 1
    encoder_df['gain_on'] = encoder_df['gain_on'].ffill()
    return encoder_df


def get_sweep_acquired_behavior_data(h5_file_path, pulley_diameter=0.31, joystick_length=0.13, get_timestamps=False,
                                     joystick_ticks=4096, pulley_ticks=2048, only_changes=True) -> object:
    h5_file_name_prefix = re.sub('\.h5$', '', os.path.basename(h5_file_path))
    h5_file_name_prefix = re.sub('\.txt$', '', h5_file_name_prefix)
    experiment_date = h5_file_name_prefix.split('_')[-2]
    experiment_date = datetime.strptime(experiment_date, '%Y-%m-%d')

    encoder_df = fast_load_wavesurfer_binary_into_pandas(h5_file_path, only_changes=only_changes)
    encoder_df, is_continuous = is_continuous_recording_with_sweeps(encoder_df)
    if is_continuous:
        encoder_df = change_sweep_times_for_continuous_recordings_with_triggers_behavior(encoder_df)
        encoder_df = change_sweep_numbers_for_continuous_recordings_with_triggers_behavior(encoder_df)
        encoder_df = change_gains_for_continuous_recordings_with_triggers_behavior(encoder_df)
    else:
        encoder_df.loc[encoder_df.sweep.diff() > 0, 'trigger_start'] = 1
        encoder_df = encoder_df.copy()
        encoder_df.at[0, 'trigger_start'] = 1
        encoder_df['sweep_start_time'] = np.nan
    encoder_df = zero_sweeped_encoder_sweeps(encoder_df, is_continuous)
    encoder_df = scale_encoder_data(encoder_df, pulley_diameter, joystick_length, joystick_ticks, pulley_ticks,
                                    round_sweep_time_to=0.001, acq4=True)

    inverse_encoder_start_date = datetime.strptime('2023-09-11', '%Y-%m-%d')
    inverse_encoder_end_date = datetime.strptime('2024-02-10', '%Y-%m-%d')
    if inverse_encoder_start_date <= experiment_date:
        encoder_df['xPos'] *= -1
    if experiment_date >= inverse_encoder_end_date:
        encoder_df['motorPos'] *= -1

    encoder_df['sweep_start'] = encoder_df['sweep'].diff()
    if 'gain_on' not in encoder_df.columns:
        encoder_df['gain_on'] = 1
        fix_gains = True
    if is_continuous:
        #remove the offset created if the recording starts in the middle of a pull
        encoder_df['motorPos'] -= encoder_df['motorPos'].iloc[:5000].quantile(0.1)
        encoder_df['xPos'] -= encoder_df['xPos'].iloc[:5000].quantile(0.1)
    encoder_df['acquisition'] = 'ws'
    encoder_df = mark_zero_crossings_per_sweep(encoder_df)
    encoder_df = mark_false_starts_and_move_tests2(encoder_df)
    encoder_df = mark_sweep_number_since_gain_change(encoder_df, exclude_false_starts=True)
    encoder_df = mark_control_pulls(encoder_df)
    encoder_df.loc[encoder_df['gain_on'] == -1, 'Condition'] = 'Control'
    encoder_df.loc[encoder_df['move_test'] == 2, 'Condition'] = 'Water Port'
    encoder_df.loc[(encoder_df['gain_on'] == 0) & (encoder_df['move_test'] != 2), 'Condition'] = 'Gain Down'
    encoder_df.loc[(encoder_df['gain_on'] == 1) & (encoder_df['move_test'] != 2), 'Condition'] = 'Gain Up'
    encoder_df.loc[encoder_df['catch_trial'] == 1, 'Condition'] = 'Catch Trial'
    encoder_df['Pull'] = encoder_df['pull_number_since_gain']
    encoder_df['Velocity (cm/s)'] = encoder_df['xV_smooth']
    encoder_df['Velocity (cm/s)'] *= 100
    # add_sweep_time_bins(encoder_df, bin_width=0.005)
    encoder_df['Rewarded'] = 0
    if 'reward_pumping' in encoder_df.columns:
        encoder_df = mark_rewarded_sweeps(encoder_df)
    else:
        encoder_df['Reward'] = 'Presumed Delivered'
    return encoder_df


def interpolate_behavior(behavior_df, col='xPos', sweep_start=-1., sweep_end=2., timestep=0.001):
    data = behavior_df.copy()
    data = data.loc[data.sweep_time.between(sweep_start, sweep_end)]
    new_timebase = np.arange(sweep_start, sweep_end, timestep)
    interpolated_pulls = pd.DataFrame(index=new_timebase)
    for sweep in data.sweep.unique():
        try:
            pull_data = data[data.sweep == sweep][[col, 'sweep_time']]
            pull_data = pull_data.loc[pull_data[col].diff() != 0]
            f = interp1d(pull_data['sweep_time'], pull_data[col], kind='slinear', fill_value='extrapolate')
            interpolated_pulls[sweep] = f(new_timebase)
            interpolated_pulls = interpolated_pulls.copy()
        except ValueError:
            # print(f'no {col} data for sweep {sweep}, probably a move test artifact')
            pass
    return interpolated_pulls


def get_velocities(df, col='xPos', win_start=0, win_end=0.1, timestep=0.001):
    df = df.copy()
    df[col] *= 100  # convert to cm
    interpolated_pulls = interpolate_behavior(df, col=col, sweep_start=win_start - 0.5, sweep_end=win_end + 0.8,
                                              timestep=timestep)
    velocity_df = interpolated_pulls.diff(axis=0).rolling(int(0.01 / timestep), center=True).mean()
    velocity_df /= timestep  # convert to s
    velocity_df = velocity_df[win_start: win_end]
    return velocity_df


def get_endpoints(df, which='xPos', peak_v_time=75):
    block_dict = df.set_index('gain_count').sweep.drop_duplicates().reset_index().set_index('sweep').to_dict()
    df = df[(df.Condition != 'Water Port') & (~df.Pull.isin([-1, 0]))]
    x = interpolate_behavior(df, col=which, sweep_start=0, sweep_end=0.2, timestep=0.001)
    x.index = np.round(x.index, 3)
    v = get_velocities(df, col=which, win_start=0, win_end=0.2, timestep=0.001)
    v.index = np.round(v.index, 3)
    sweeps = x.columns.intersection(v.columns)
    endpoints = pd.DataFrame(
        [[v[sweep].max(), x[sweep][v[sweep].idxmax()], x[sweep].loc[:0.1].sum()] for sweep in sweeps], index=sweeps,
        columns=['peak_v', 'x_peak_v', 'x_sum'])
    endpoints[['Condition', 'Pull']] = df.groupby('sweep')[['Condition', 'Pull']].first()
    endpoints.Pull = endpoints.Pull.astype(int)
    endpoints.reset_index(inplace=True)
    endpoints.rename(columns={'index': 'sweep'}, inplace=True)
    endpoints.x_peak_v *= 1000  # convert to mm
    endpoints.x_sum *= 1000  # convert to mm
    if which == 'xPos':
        endpoints.loc[endpoints.Condition == 'Gain Up', 'x_sum'] *= 2
    endpoints['block'] = endpoints.sweep.map(block_dict['gain_count'])
    return endpoints


def all_trials(encoder_df, window=[0, 0.25], n_trials=5, which='xPos', plot_error=False):
    conditions = []
    for trial in range(1, n_trials + 1):
        conditions.append([(encoder_df.Condition == 'Gain Up') & (encoder_df.pull_number_since_gain == trial),
                           (encoder_df.Condition == 'Gain Down') & (encoder_df.pull_number_since_gain == trial)])
    all_trials_df = interpolate_behavior(encoder_df[encoder_df.Condition == 'Control'], col=which,
                                         sweep_start=window[0], sweep_end=window[1], timestep=0.001)
    all_trials_df = all_trials_df * 1000  # convert to mm
    all_trials_df.index *= 1000  # convert to ms
    all_trials_df = pd.melt(all_trials_df.reset_index(), id_vars='index', var_name='trial', value_name=which)
    mean_control = all_trials_df.drop(columns='trial').groupby('index').mean().reset_index()
    mean_control.rename(columns={which: 'control'}, inplace=True)
    all_trials_df['Condition'] = 'Control'
    all_trials_df['Pull'] = 1
    for condition, pull in zip(conditions, range(1, n_trials + 1)):
        down_df = interpolate_behavior(encoder_df[condition[0]], col=which, sweep_start=window[0], sweep_end=window[1],
                                       timestep=0.001)
        down_df = down_df * 1000  # convert to mm
        down_df.index *= 1000  # convert to ms
        down_df = pd.melt(down_df.reset_index(), id_vars='index', var_name='trial', value_name=which)
        if plot_error:
            down_df = pd.merge(down_df, mean_control, on='index')
            down_df[which] -= down_df.control
        down_df['Condition'] = 'Gain Down'
        down_df['Pull'] = pull
        up_df = interpolate_behavior(encoder_df[condition[1]], col=which, sweep_start=window[0], sweep_end=window[1],
                                     timestep=0.001)
        up_df = up_df * 1000  # convert to mm
        if plot_error:
            up_df *= 2
        up_df.index *= 1000  # convert to ms
        up_df = pd.melt(up_df.reset_index(), id_vars='index', var_name='trial', value_name=which)
        if plot_error:
            up_df = pd.merge(up_df, mean_control, on='index')
            up_df[which] -= up_df.control
        up_df['Condition'] = 'Gain Up'
        up_df['Pull'] = pull
        all_trials_df = pd.concat([all_trials_df, down_df, up_df], ignore_index=True)
    return all_trials_df


def prepare_condition(df, condition, pull, window, block_dict):
    subset = df[(df.Condition == condition) & (df.pull_number_since_gain == pull)]
    interp = interpolate_behavior(subset, col='xPos',
                                  sweep_start=window[0],
                                  sweep_end=window[1],
                                  timestep=0.001)

    tidy = pd.melt(interp.reset_index(),
                   id_vars='index',
                   var_name='trial',
                   value_name='xPos')
    tidy = tidy.rename(columns={'index': 'sweep_time'})
    tidy['sweep_time'] = (tidy['sweep_time'] * 1000).astype(int)
    tidy['xPos'] *= 1000
    tidy['block'] = tidy.trial.map(block_dict['gain_count'])
    tidy.set_index(['block', 'sweep_time'], inplace=True)
    return tidy


def compare_pairs(df_a, df_b, shift_a=0, shift_b=0, p_val_thresh=0.05, min_time=30):
    a_idx = pd.MultiIndex.from_tuples([(i[0] + shift_a, i[1]) for i in df_a.index],
                                      names=df_a.index.names)
    b_idx = pd.MultiIndex.from_tuples([(i[0] + shift_b, i[1]) for i in df_b.index],
                                      names=df_b.index.names)
    df_a_shift = df_a.set_index(a_idx)
    df_b_shift = df_b.set_index(b_idx)
    merged = pd.merge(df_a_shift, df_b_shift, left_index=True, right_index=True).dropna()
    merged['xPos_diff'] = merged.xPos_y - merged.xPos_x

    results = []
    for t in merged.reset_index().sweep_time.unique():
        x = merged.xs(t, level='sweep_time').xPos_y
        y = merged.xs(t, level='sweep_time').xPos_x
        if len(x) > 0 and len(y) > 0:
            stat, p = stats.wilcoxon(x, y)
            results.append((t, merged[merged.index.get_level_values('sweep_time') == t]['xPos_diff'].mean(), stat, p))

    out = pd.DataFrame(results, columns=['sweep_time', 'xPos_diff', 'W', 'p'])
    out = out[out.sweep_time > min_time]

    sep_time = np.nan
    sig = out[out['p'] < p_val_thresh]
    if not sig.empty:
        sep_time = sig.iloc[0].sweep_time
    return out, sep_time


def feedback_delay_times(df, window=[-0.0125, 0.25], p_val_thresh=0.05):
    conditions = {
        "up1": ('Gain Up', 1),
        "down5": ('Gain Down', 5),
        "down1": ('Gain Down', 1),
        "up5": ('Gain Up', 5)
    }
    block_dict = df.set_index('gain_count').sweep.drop_duplicates().reset_index().set_index('sweep').to_dict()
    prepared = {name: prepare_condition(df, cond, pull, window, block_dict)
                for name, (cond, pull) in conditions.items()}

    up1_down5, up1_down5_sep_time = compare_pairs(prepared['up1'], prepared['down5'], shift_a=-1,
                                                  p_val_thresh=p_val_thresh)
    up5_down1, up5_down1_sep_time = compare_pairs(prepared['up5'], prepared['down1'], shift_b=-1,
                                                  p_val_thresh=p_val_thresh)
    up1_down1, up1_down1_sep_time = compare_pairs(prepared['up1'], prepared['down1'], shift_a=+1,
                                                  p_val_thresh=p_val_thresh)
    up5_down5, up5_down5_sep_time = compare_pairs(prepared['up5'], prepared['down5'], shift_a=+1,
                                                  p_val_thresh=p_val_thresh)

    feedback_delay = np.nanmean([up1_down5_sep_time, up5_down1_sep_time])
    feedforward_delay = np.nanmean([up1_down1_sep_time, up5_down5_sep_time])
    return feedback_delay, feedforward_delay


def read_timestamp_of_recording(desc):
    keyword_idx = desc.find('epoch')
    date_string = re.split('\[|\]', desc[keyword_idx:])
    date_idv = date_string[1].split()
    unix_start_time = int(
        datetime(int(date_idv[0]), int(date_idv[1]), int(date_idv[2]), int(date_idv[3]), int(date_idv[4]),
                 int(float(date_idv[5]))).strftime('%s'))
    keyword_idx = desc.find('frameTimestamps_sec')
    split_string = re.split('=|\n', desc[keyword_idx:])
    frameTimestamps = float(split_string[1])

    keyword_idx = desc.find('acqTriggerTimestamps_sec')
    split_string = re.split('=|\n', desc[keyword_idx:])
    acqTriggerTimestamps = float(split_string[1])

    keyword_idx = desc.find('frameNumberAcquisition')
    split_string = re.split('=|\n', desc[keyword_idx:])
    frameNumberAcquisition = int(split_string[1])

    keyword_idx = desc.find('acquisitionNumbers')
    split_string = re.split('=|\n', desc[keyword_idx:])
    acquisitionNumbers = int(split_string[1])

    keyword_idx = desc.find('frameNumbers')
    split_string = re.split('=|\n', desc[keyword_idx:])
    frameNumbers = int(split_string[1])

    unixFrameTime = unix_start_time + frameTimestamps
    return (
        {
            'frameNumbers': frameNumbers,
            'frameNumberAcquisition': frameNumberAcquisition,
            'acquisitionNumbers': acquisitionNumbers,
            'unix_start_time': unix_start_time,
            'unixFrameTime': unixFrameTime,
            'frameTimestamps': frameTimestamps,
            'acqTriggerTimestamps': acqTriggerTimestamps
        }
    )


def extract_scanimage_timestamps_from_list_of_tifs(image_paths, force_redo=False, label_previous_acq=True):
    time_stamps = []
    image_paths = sorted(np.unique(image_paths))
    for image_path in image_paths:
        print(image_path)
        reader = ScanImageTiffReader(image_path)
        frames = reader.shape()[0]
        for frame in range(frames):
            desc = reader.description(frame)
            stamp = read_timestamp_of_recording(desc)
            time_stamps.append(stamp)
        reader.close()
    df = pd.DataFrame(time_stamps)
    df['frame_time_since_trigger'] = df.frameTimestamps - df.acqTriggerTimestamps
    reader = ScanImageTiffReader(image_paths[0])
    meta = reader.metadata()
    active_chan_index = meta.find('channelsActive')
    split_string = re.split('=|\n', meta[active_chan_index:])
    active_channels = [int(s) for s in split_string[1] if s.isdigit()]
    df['chan'] = active_channels[0]
    df.loc[df.frameNumberAcquisition.diff() == 0, 'chan'] = int(active_channels[-1])
    if label_previous_acq:
        df['previous_acq'] = 0
        df.loc[df.acquisitionNumbers.shift(-1).diff() < 0, 'previous_acq'] = 1
        df['previous_acq'] = df['previous_acq'][::-1].cumsum()
    return df


def get_tif_timestamps_associated_with_an_h5(h5_file_path, force_redo=False, drop_previous_acq=True, save_pickle=True):
    ca_path = os.path.dirname(h5_file_path)
    h5_file_name_prefix = re.sub('\.h5$', '', os.path.basename(h5_file_path))
    timing_save_name = ca_path + '/' + h5_file_name_prefix + '_frametimes.pkl'
    try:
        if not force_redo:
            all_frame_times = pd.read_pickle(timing_save_name)
            if 'previous_acq' not in all_frame_times.columns:
                pd.read_pickle('a nonexistent file')
            else:
                print("     loaded", timing_save_name)
        else:
            force_a_redo_hack = 'a nonexistent file'
            pd.read_pickle(force_a_redo_hack)
    except FileNotFoundError:
        print("couldn't find", timing_save_name)
        fp = Path(h5_file_path)
        h5_tifs = np.sort([str(tif) for tif in fp.parents[0].glob('*.tif')])
        all_frame_times = extract_scanimage_timestamps_from_list_of_tifs(h5_tifs, force_redo=force_redo)
    if drop_previous_acq:
        if all_frame_times.previous_acq.max() > 0:
            start_of_current_acq = \
            all_frame_times.loc[all_frame_times['previous_acq'].diff() < 0, 'frameTimestamps'].values[0]
            all_frame_times.frameTimestamps += start_of_current_acq
        all_frame_times = all_frame_times.loc[all_frame_times['previous_acq'] == 0].reset_index(drop=True)
    return all_frame_times


def fast_load_suite2p_data_for_wavesurfer_h5(h5_file_path, region='dendrites', exclude_previous_acq=True,
                                             tau=0.25, fs=30.0, sig_baseline=10.0):
    h5_folder_path = os.path.dirname(h5_file_path)
    suite2p_folder_name = "suite2p_" + re.sub('\.h5$', '', os.path.basename(h5_file_path))
    suite2p_path = h5_folder_path + '/' + suite2p_folder_name + region

    all_frame_times = get_tif_timestamps_associated_with_an_h5(h5_file_path, drop_previous_acq=exclude_previous_acq)
    f_df = all_frame_times.loc[all_frame_times['chan'] == 1].reset_index(drop=True).copy()
    f = np.load(suite2p_path + '/plane0/F.npy')
    f_names = ['cell_' + str(cell) + '_fluo_chan1' for cell in np.arange(f.shape[0])]
    spks = np.load(suite2p_path + '/plane0/spks.npy')
    spks_names = ['cell_' + str(cell) + '_spikes' for cell in np.arange(f.shape[0])]
    iscell = np.load(suite2p_path + '/plane0/iscell.npy')
    bad_cells = ['cell_' + str(cell) for cell in np.arange(iscell.shape[0]) if iscell[cell, 0] == 0]

    f_df = f_df.join(pd.DataFrame(f.T, columns=f_names), how='outer')
    f_df = f_df.join(pd.DataFrame(spks.T, columns=spks_names), how='outer')
    f_df['sweep'] = f_df['acquisitionNumbers']
    f_df['sweep_time'] = f_df['frame_time_since_trigger']

    f_df.drop(columns=[cell + '_spikes' for cell in bad_cells], inplace=True)
    f_df.drop(columns=[cell + '_fluo_chan1' for cell in bad_cells], inplace=True)
    if exclude_previous_acq:
        f_df = f_df.loc[f_df['previous_acq'] == 0].reset_index(drop=True)
    return f_df


def retime_suite2p_to_pulls_for_continuous_recordings_with_triggers(ca_df, pre_pull_buffer=1.):
    frame_time = ca_df.frameTimestamps.diff().median()
    pre_pull_frames = int(pre_pull_buffer / frame_time)
    return ca_df


def get_pull_cadf(h5_file_path, region='dendrites', pre_pull_buffer=1.):
    encoder_df = get_sweep_acquired_behavior_data(h5_file_path)
    ca_df = fast_load_suite2p_data_for_wavesurfer_h5(h5_file_path, region=region)
    frame_time = ca_df.frameTimestamps.diff().median()
    pre_pull_frames = int(pre_pull_buffer / frame_time)
    ca_df['pull'] = 0
    ca_df.loc[ca_df.shift(-pre_pull_frames)['acquisitionNumbers'].diff() > 0, ['pull']] = 1
    ca_df.loc[ca_df['acquisitionNumbers'].diff() > 0, ['pull_thresh']] = 1
    ca_df['pull'] = ca_df['pull'].cumsum()
    ca_df.rename(columns={'frameNumberAcquisition': 'original_frameNumberAcquisition',
                          'acquisitionNumbers': 'original_acquisitionNumbers',
                          'frame_time_since_trigger': 'original_frame_time_since_trigger'},
                 inplace=True)
    ca_df['frameNumberAcquisition'] = 0
    ca_df.loc[ca_df['pull'].diff() > 0, 'frameNumberAcquisition'] = 1
    ca_df.loc[ca_df['pull'] > 0, 'frameNumberSincePull'] = 1
    ca_df.loc[ca_df['pull'].diff() > 0, 'frameNumberSincePull'] = -pre_pull_frames
    for pull in ca_df['pull'].unique():
        ca_df.loc[ca_df['pull'] == pull, 'frameNumberAcquisition'] = ca_df.loc[
            ca_df['pull'] == pull, 'frameNumberAcquisition'].cumsum()
        ca_df.loc[ca_df['pull'] == pull, 'frameNumberSincePull'] = ca_df.loc[
            ca_df['pull'] == pull, 'frameNumberSincePull'].cumsum()
    ca_df['frameNumberSincePull'] = ca_df['frameNumberSincePull'].fillna(0).astype('int')
    ca_df['acquisitionNumbers'] = ca_df['pull']
    ca_df['frame_time_since_trigger'] = ca_df['frameNumberSincePull'] * np.round(frame_time, 3)
    ca_df['sweep'] = ca_df['pull']
    ca_df['acquisitionNumbers'] = ca_df['sweep']
    sweeps = ca_df['acquisitionNumbers'].unique()
    for sweep in sweeps[sweeps != 0]:
        encoder_sweep = encoder_df.loc[encoder_df['sweep_num'] == sweep]
        if len(encoder_sweep) > 0:
            ca_df.loc[ca_df['acquisitionNumbers'] == sweep, ['gain_on',
                                                             'move_test',
                                                             'false_start',
                                                             'pull_number_since_gain',
                                                             'Condition',
                                                             'Rewarded']] = [encoder_sweep['gain_on'].max(),
                                                                              encoder_sweep['move_test'].max(),
                                                                              encoder_sweep['false_start'].max(),
                                                                              encoder_sweep[
                                                                                  'pull_number_since_gain'].max(),
                                                                              encoder_sweep.iloc[0]['Condition'],
                                                                             encoder_sweep['Rewarded'].max(),
                                                                              ]
    return ca_df


def get_spikes_from_ca_df(ca_df, remove_doublets=True):
    F_cols = [col for col in ca_df.columns if '_fluo_chan1' in col]
    F_med = ca_df[F_cols].median().values
    spike_cols = [col for col in ca_df.columns if
                  '_spikes' in col and not '_spikes_chan2' in col and not '_spikes_red_subtract' in col]
    spks = ca_df[spike_cols].copy()
    spks_norm = spks.divide(F_med, axis='columns')
    spks_norm = spks_norm.divide(spks_norm.quantile(.99).values, axis='columns')
    spikes = spks_norm > 0.4
    if remove_doublets:
        spikes_np = spikes.to_numpy()
        shifted_spikes = np.roll(spikes_np, shift=-1, axis=0)
        consecutive_spikes = spikes_np & shifted_spikes
        spikes_np[consecutive_spikes] = False
        spikes = pd.DataFrame(spikes_np, columns=spike_cols)
    spikes = spikes.astype(np.bool_)
    return spikes


def get_s2p_npy_path(h5_file_path, npy_kind):
    h5_folder_path = os.path.dirname(h5_file_path)
    suite2p_folder_name = "suite2p_" + re.sub('\.h5$', '', os.path.basename(h5_file_path))
    suite2p_path = h5_folder_path + '/' + suite2p_folder_name + 'dendrites'
    npy_path = suite2p_path + '/' + 'plane0/' + npy_kind + '.npy'
    return npy_path


def remove_not_iscell_cols(ca_df, h5_file_path):
    col_kinds = ['_spikes', '_fluo_chan1', '_fluo_chan2', '_neu']
    cols = [col for col in ca_df.columns if any([col_kind in col for col_kind in col_kinds])]
    iscell_path = get_s2p_npy_path(h5_file_path, 'iscell')
    iscell = np.load(iscell_path)
    not_iscell_cols = [col for col in cols if iscell[int(re.search(r'(\d+)', col).group(1)), 0] == 0]
    ca_df.drop(not_iscell_cols, axis=1, inplace=True)
    return ca_df


def load_ca_df(h5_file_path, remove_doublets=True, get_spikes=True, remove_not_iscell=True):
    ca_df = get_pull_cadf(h5_file_path)
    if get_spikes:
        spike_cols = [col for col in ca_df.columns if '_spikes' in col and not '_spikes_chan2' in col]
        spikes = get_spikes_from_ca_df(ca_df, remove_doublets=remove_doublets)
        ca_df[spike_cols] = spikes
    if remove_not_iscell:
        ca_df = remove_not_iscell_cols(ca_df, h5_file_path)
    ca_df = ca_df[ca_df.pull_number_since_gain != 0]
    return ca_df


def example_cell_sweeps(ca_df, example_cell, pull=1, condition='Water Port', window=(-15, 30)):
    fluo_col = 'cell_' + str(example_cell) + '_fluo_chan1'
    spike_col = 'cell_' + str(example_cell) + '_spikes'
    sweep_df = ca_df[(ca_df['frameNumberSincePull'].between(window[0], window[1])) & (ca_df.Condition == condition) &
                     (ca_df.pull_number_since_gain == pull)][
        ['acquisitionNumbers', 'frameNumberSincePull', fluo_col, spike_col]].copy()
    psth_df = pd.melt(sweep_df.copy(), id_vars=['acquisitionNumbers', 'frameNumberSincePull'], var_name='cell',
                      value_name='spike')
    psth_df = psth_df[(psth_df['spike'] == 1)]
    return sweep_df, psth_df


def chi2_test(table1, table2):
    contingency_table = [table1, table2]
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table, correction=True)
    return chi2, p, dof, expected


def example_heatmaps(ca_df, conditions=['Gain Up', 'Gain Down'], pulls=[1, 5], window=(-15, 30), rolling_mean=True):
    spike_cols = [col for col in ca_df.columns if '_spikes' in col]
    extra_cols = ['Condition', 'pull_number_since_gain', 'acquisitionNumbers', 'frameNumberSincePull']
    spike_df = ca_df[spike_cols + extra_cols]
    if rolling_mean:
        spike_df[spike_cols] = spike_df[spike_cols].rolling(3, center=True).mean()
    spike_df = spike_df[(spike_df['frameNumberSincePull'].between(window[0], window[1]))]
    heatmaps = []
    for condition in conditions:
        for pull in pulls:
            heatmap_df = spike_df[
                (spike_df['Condition'] == condition) & (spike_df['pull_number_since_gain'] == pull)].drop(
                columns=['Condition', 'pull_number_since_gain', 'acquisitionNumbers']).copy()
            heatmap_df = heatmap_df.set_index('frameNumberSincePull')
            heatmap_df.index = heatmap_df.index.astype(int)
            heatmap_df = heatmap_df.groupby(['frameNumberSincePull']).sum() / heatmap_df.groupby(
                ['frameNumberSincePull']).count() / 0.033
            heatmap_df = pd.to_numeric(heatmap_df.stack(), errors='coerce').unstack()
            heatmaps += [heatmap_df]
    return heatmaps


def get_clusters(ca_df, conditions=['Gain Up', 'Gain Down'], pulls=[1, 5], frame_window=(-10, 15), n_iterations=32,
                 force_k=None):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    spike_cols = [col for col in ca_df.columns if '_spikes' in col]
    ca_df[spike_cols] = ca_df[spike_cols].rolling(3, center=True).mean()
    cluster_data = ca_df[(ca_df.Condition.isin(conditions)) & (ca_df.pull_number_since_gain.isin(pulls)) & ca_df[
        'frameNumberSincePull'].between(frame_window[0], frame_window[1])][
        spike_cols + ['Condition', 'pull_number_since_gain', 'frameNumberSincePull']]
    cluster_data = cluster_data.groupby(
        ['Condition', 'pull_number_since_gain', 'frameNumberSincePull']).sum() / cluster_data.groupby(
        ['Condition', 'pull_number_since_gain', 'frameNumberSincePull']).count() / 0.033
    cluster_data = np.sqrt(cluster_data)
    cluster_data = (cluster_data - cluster_data.mean()) / cluster_data.std()
    cluster_data = cluster_data.T
    if force_k is None:
        k_values = range(2, 15)
        optimal_ks = []
        for i in range(n_iterations):
            silhouette_scores = []
            for k in k_values:
                kmeans = KMeans(n_clusters=k, n_init='auto').fit(cluster_data)
                cluster_labels = kmeans.labels_
                silhouette_avg = silhouette_score(cluster_data, cluster_labels)
                silhouette_scores.append(silhouette_avg)
            optimal_k = k_values[silhouette_scores.index(max(silhouette_scores))]
            optimal_ks.append(optimal_k)
        from scipy.stats import mode
        optimal_k = mode(optimal_ks).mode
        print('mode (optimal k) =', optimal_k)
        print('count/total = ' + str(mode(optimal_ks).count) + '/' + str(len(optimal_ks)))
    else:
        optimal_k = force_k

    if optimal_k > 1:
        sum_of_squares_list = []
        for _ in range(n_iterations):
            seed = np.random.randint(0, 1000)
            kmeans = KMeans(n_clusters=optimal_k, random_state=seed, n_init='auto').fit(cluster_data)
            sum_of_squares = kmeans.inertia_
            sum_of_squares_list.append(sum_of_squares)
        best_iteration = np.argmin(sum_of_squares_list)
        best_seed = np.random.seed(best_iteration)
        best_kmeans = KMeans(n_clusters=optimal_k, random_state=best_seed).fit(cluster_data)
        cluster_labels = best_kmeans.labels_
    else:
        cluster_labels = np.ones(len(cluster_data))
    index = cluster_data.index
    cluster_labels = pd.DataFrame(cluster_labels, index=index, columns=['Cluster'])
    cluster_labels.Cluster = cluster_labels.Cluster.astype(int)
    return cluster_labels


def cluster_rate_vs_endpoint(encoder_df, ca_df, crit_period=range(0, 10),
                             n_iterations=32, force_k=None, roll=True, sub_control=True, spike_rate_period=range(2, 5)):
    spike_cols = [col for col in ca_df.columns if '_spikes' in col]
    extra_cols = ['Condition', 'pull_number_since_gain', 'acquisitionNumbers', 'frameNumberSincePull']
    spike_df = ca_df[ca_df.pull_number_since_gain.between(1, 5)][spike_cols + extra_cols]
    if roll:
        spike_df[spike_cols] = spike_df[spike_cols].rolling(3, center=True).mean().dropna()
    cluster_labels = get_clusters(ca_df=ca_df, n_iterations=n_iterations, force_k=force_k)
    endpoints = get_endpoints(encoder_df)
    cols = ['x_peak_v', 'x_sum']
    control_endpoint = endpoints[endpoints['Condition'].isin(['Control'])][cols].mean()
    endpoints['x_sum'] -= control_endpoint['x_sum']
    if sub_control:
        endpoints['x_peak_v'] -= control_endpoint['x_peak_v']
    df = spike_df[spike_df['frameNumberSincePull'].isin(crit_period)]
    df = df.groupby(['Condition', 'pull_number_since_gain', 'frameNumberSincePull', 'acquisitionNumbers']).mean().T
    df['Cluster'] = cluster_labels.Cluster
    df = df.groupby('Cluster').mean()
    df = df.T.reset_index().pivot_table(index=['Condition', 'pull_number_since_gain', 'acquisitionNumbers'],
                                        columns='frameNumberSincePull',
                                        values=cluster_labels.Cluster.unique())
    if sub_control:
        for cluster in df.columns.get_level_values(0).unique():
            mean_control = df.loc[('Control', 1), cluster].mean(axis=0)
            hmmm = df.loc[:, cluster].sub(mean_control, axis=1)
            df.loc[:, cluster] = hmmm.values
    mean_control = df.median(axis=0)

    pca_df = pd.DataFrame()
    for cluster, group in df.groupby(axis=1, level='Cluster'):
        group[(cluster, 'spike_rate')] = group.loc[:, pd.IndexSlice[cluster, spike_rate_period]].mean(axis=1)
        group = group.drop(crit_period, axis=1, level=1)
        pca_df = pd.concat([pca_df, group], axis=1)
    pca_df.columns = pca_df.columns.droplevel('frameNumberSincePull')
    plot_df = pd.melt(pca_df.reset_index(), id_vars=['Condition', 'pull_number_since_gain', 'acquisitionNumbers'],
                      value_name='spike_rate')
    plot_df.pull_number_since_gain = plot_df.pull_number_since_gain.astype(int)
    plot_df = pd.merge(plot_df, endpoints[['sweep', 'block'] + cols], left_on='acquisitionNumbers', right_on='sweep')
    plot_df = plot_df.drop(columns=['acquisitionNumbers', 'sweep', 'block']).groupby(
        ['Condition', 'pull_number_since_gain', 'Cluster']).mean().reset_index()
    return plot_df

def spikes_above_threshold(ca_df, threshold=100, drop_doublets=True):
    spike_cols = [col for col in ca_df.columns if '_spikes' in col]
    filtered_spikes = ca_df[spike_cols][ca_df[spike_cols] > threshold]
    filtered_spikes = (filtered_spikes > 0).astype(int)
    if drop_doublets:
        temp = filtered_spikes[filtered_spikes.diff().fillna(0) != 0]
        return ca_df.drop(columns=spike_cols).join(temp)
    else:
        return ca_df.drop(columns=spike_cols)


def get_clusters_bs(h5_file_path, ca_df=None, standardise=True, window=(0, 10), force_k=None, force_redo=True,
                     psth_clustering=True, cluster_on=(['Gain Up', 'Gain Down'], (1, 2, 3, 4, 5)), spike_percentile=100,
                     threshold_spikes=True):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    print(f"Clustering on{cluster_on} for window {window}")
    if ca_df is None:
        ca_df = get_pull_cadf(h5_file_path=h5_file_path)
        if threshold_spikes:
            ca_df = spikes_above_threshold(ca_df)
    spike_cols = [col for col in ca_df.columns if
                          '_spikes' in col and not '_spikes_chan2' in col and not '_spikes_red_subtract' in col]

    frame_number_var = 'frameNumberSincePull' if 'frameNumberSincePull' in ca_df.columns else 'frameNumberAcquisition'

    data = ca_df[ca_df[frame_number_var].between(window[0], window[1])][
        spike_cols + ['Condition', 'pull_number_since_gain', frame_number_var]]
    if cluster_on:
        def build_condition_mask(series, keys, case_sensitive=True):
            s = series.astype('string')
            if not case_sensitive:
                s = s.str.lower()
                keys = [k.lower() for k in keys]

            mask = pd.Series(False, index=s.index)
            for k in keys:
                if k.endswith('*'):
                    prefix = k[:-1]
                    mask |= s.str.startswith(prefix, na=False)
                else:
                    mask |= s.eq(k)
            return mask

        conds, pulls = cluster_on
        mask_cond = build_condition_mask(data['Condition'], conds)
        mask_pulls = data['pull_number_since_gain'].isin(pulls)
        data = data.loc[mask_cond & mask_pulls].copy()
        print(f"Pulls {data['pull_number_since_gain'].unique()}")
        print(f"Conditions {data['Condition'].unique()}")
        print(f"Window {data[frame_number_var].min()} to {data[frame_number_var].max()}")
    if psth_clustering:
        print('making PSTHs')
        data = data.groupby(['Condition', 'pull_number_since_gain', frame_number_var]).sum() / data.groupby(
            ['Condition', 'pull_number_since_gain', frame_number_var]).count()
    else:
        data = data.groupby(['Condition', 'pull_number_since_gain', frame_number_var]).mean()
    if standardise:
        data = (data - data.mean()) / data.std()
        print('standardized')

    data = data.reset_index(drop=True)[spike_cols].T
    data.dropna(inplace=True)

    if force_k is None:
        num_samples = data.shape[0]
        max_k = min(15, num_samples - 1)  # Ensure max_k does not exceed the number of samples
        k_values = range(2, max_k + 1)  # Ensure k starts from 2 and goes up to max_k
        silhouette_scores = []

        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
            cluster_labels = kmeans.labels_

            # Only compute silhouette score if the number of clusters is less than the number of samples
            if k < num_samples:
                silhouette_avg = silhouette_score(data, cluster_labels)
                silhouette_scores.append(silhouette_avg)
            else:
                silhouette_scores.append(-1)  # Append a placeholder for invalid k values

        optimal_k = k_values[silhouette_scores.index(max(silhouette_scores))]
        print('Optimal k =', optimal_k)
    else:
        optimal_k = force_k

    best_kmeans = KMeans(n_clusters=optimal_k, random_state=42).fit(data)

    cluster_labels = best_kmeans.labels_
    index = data.index
    cluster_labels = pd.DataFrame(cluster_labels, index=index, columns=['Cluster'])
    cluster_labels.Cluster += 1
    cluster_labels.reset_index(names='spikes', inplace=True)
    cluster_labels['roi'] = cluster_labels['spikes'].str.extract('(\d+)').astype(int)
    return cluster_labels


def interpolate_behavior_general(behavior_df, columns, sweep_start=-1., sweep_end=2., timestep=0.001, vertical=False,
                                 verbose=False):
    data = behavior_df.copy()
    data = data.loc[data.sweep_time.between(sweep_start, sweep_end)]
    new_timebase = np.arange(sweep_start, sweep_end, timestep)
    interpolated_pulls = {col: pd.DataFrame(index=new_timebase) for col in columns}

    for sweep in data.sweep.unique():
        try:
            for col in columns:
                pull_data = data[data.sweep == sweep][[col, 'sweep_time']]
                pull_data = pull_data.loc[pull_data[col].diff() != 0]
                first_val = pull_data[col].iloc[0]
                first_time = pull_data['sweep_time'].iloc[0]
                f = interp1d(pull_data['sweep_time'], pull_data[col], kind='slinear', fill_value='extrapolate')
                interpolated_pulls[col][sweep] = f(new_timebase)
                interpolated_pulls[col].loc[:first_time, sweep] = first_val
                interpolated_pulls[col] = interpolated_pulls[col].copy()
        except ValueError:
            if verbose:
                print(f'Error: missing behavior data for sweep {sweep}, possibly a move test artifact')
            pass

    if vertical:
        melted_data = []
        for col in columns:
            col_data = interpolated_pulls[col]
            col_data = col_data.reset_index(names='sweep_time').melt(id_vars='sweep_time', var_name='sweep', value_name=col)
            melted_data.append(col_data)

        merged_data = melted_data[0]
        for col_data in melted_data[1:]:
            merged_data = pd.merge(merged_data, col_data, on=['sweep_time', 'sweep'])

        condition_map = data.set_index('sweep')['Condition'].to_dict()
        pull_number_map = data.set_index('sweep')['pull_number_since_gain'].to_dict()
        reward_map = data.set_index('sweep')['Rewarded'].to_dict()

        merged_data['Condition'] = merged_data['sweep'].map(condition_map)
        merged_data['pull_number_since_gain'] = merged_data['sweep'].map(pull_number_map)
        merged_data['Rewarded'] = merged_data['sweep'].map(reward_map)
        for col in columns:
            merged_data[col] *= 100
        merged_data['(ms)'] = merged_data['sweep_time'] * 1000
        merged_data['Pull'] = merged_data['pull_number_since_gain']

        return merged_data.melt(id_vars=['(ms)', 'sweep_time', 'sweep', 'Condition', 'pull_number_since_gain', 'Pull', 'Rewarded'],
                                value_vars=columns, value_name='Position (cm)')
    else:
        return tuple(interpolated_pulls[col] for col in columns)


def get_x_positions_for_peak_velocities(df, win_start=0, win_end=0.1, timestep=0.001, rolling_mean_window=0.01,
                                        motorPos=False):
    # this returns a series with the x position at the peak velocity for each sweep; the index is the sweep number

    pos = 'motorPos' if motorPos else 'xPos'
    pos_index = 1 if motorPos else 0
    vs = get_velocities(df, win_start=win_start, win_end=win_end, timestep=timestep)
    xPos = interpolate_behavior(df, col=pos, sweep_start=win_start-0.5, sweep_end=win_end+0.8,
                                                      timestep=timestep)
    peak_v_times = vs.idxmax()
    x_at_peak_v = pd.Series({col: xPos.loc[idx, col] for col, idx in peak_v_times.items()})
    x_at_peak_v = x_at_peak_v.to_frame(name=pos)
    x_at_peak_v['peak_time'] = peak_v_times
    x_at_peak_v.reset_index(inplace=True, names='sweep')

    for sweep in x_at_peak_v['sweep'].unique():
        x_at_peak_v.loc[x_at_peak_v['sweep'] == sweep, 'Condition'] = df.loc[df['sweep'] == sweep, 'Condition'].values[0]
        x_at_peak_v.loc[x_at_peak_v['sweep'] == sweep, 'Pull'] = df.loc[df['sweep'] == sweep, 'pull_number_since_gain'].values[0]
        x_at_peak_v.loc[x_at_peak_v['sweep'] == sweep, 'gain_count'] = df.loc[df['sweep'] == sweep, 'gain_count'].values[0]
    return x_at_peak_v


def get_difference_from_control(df, variable='motorPos', sweep_start=0, sweep_end=0.135, timestep=0.001):
    df = df.loc[df['Pull'].isin([1,2,3,4,5])].copy()
    int_behavior = interpolate_behavior_general(df, [variable], sweep_start=sweep_start,
                                                                   sweep_end=sweep_end, timestep=timestep,
                                                                   vertical=True)
    int_behavior = int_behavior.loc[int_behavior['variable'] == variable].copy()
    control_mean_position = int_behavior.loc[int_behavior['Condition'] == 'Control'].groupby('(ms)')['Position (cm)'].mean()
    int_behavior['control_subtracted'] = int_behavior['Position (cm)'] - int_behavior['(ms)'].map(control_mean_position)
    int_behavior['control_subtracted'] *= 10
    difference_sums = int_behavior.groupby(['sweep', 'Condition', 'Pull'])['control_subtracted'].sum().to_frame().reset_index()
    for sweep in difference_sums['sweep'].unique():
        difference_sums.loc[difference_sums['sweep'] == sweep, 'gain_count'] = df.loc[df['sweep'] == sweep, 'gain_count'].values[0]
    return difference_sums


def get_spikes_in_cluster_and_behavior_for_each_pull(h5_file_path, ca_df=None, behav_df=None, spike_threshold=100,
                                                     drop_doublets=True, vertical_to_plot=True,
                                                     spike_window=(0.03,0.13)):
    if ca_df is None:
        ca_df = get_pull_cadf(h5_file_path)
        ca_df = spikes_above_threshold(ca_df, spike_threshold, drop_doublets)
        ca_df = ca_df.drop(columns=[col for col in ca_df.columns if '_mean_spikes' in col])
        spike_cols = [col for col in ca_df.columns if '_spikes' in col]
        ca_df[spike_cols] = ca_df[spike_cols].fillna(0)
    if behav_df is None:
        behav_df = get_sweep_acquired_behavior_data(h5_file_path)

    xs = get_difference_from_control(behav_df, sweep_start=spike_window[0],sweep_end=spike_window[1])
    x_decels = get_x_positions_for_peak_velocities(behav_df)
    xs['x_decels'] = x_decels['xPos']
    window_spikes = ca_df.loc[ca_df['frame_time_since_trigger'].between(spike_window[0], spike_window[1])]
    frames_per_sweep = window_spikes.groupby('sweep')['frameNumbers'].count().median()
    frameTime = ca_df['frameTimestamps'].diff().median()
    bline = ca_df.loc[ca_df['frame_time_since_trigger'].between(-1, -0.5)]
    bline_frames_per_sweep = bline.groupby('sweep')['frameNumbers'].count().median()

    cluster_table = get_clusters_bs(h5_file_path, ca_df=ca_df, cluster_on=(['Gain Up', 'Gain Down'], (1,2,3,4,5)),
                                 window=(0, 10), spike_percentile=100)
    for cluster in cluster_table['Cluster'].unique()[:3]:
        spike_cols = cluster_table.loc[cluster_table['Cluster'] == cluster]['spikes'].values
        spikes = window_spikes.groupby('sweep')[spike_cols].sum()
        bline_spikes = bline.groupby('sweep')[spike_cols].sum()
        mean_spikes = spikes.mean(axis=1) / (frames_per_sweep * frameTime)
        mean_bline_spikes = bline_spikes.mean(axis=1) / (bline_frames_per_sweep * frameTime)
        xs[f'Cluster_{cluster}_mean_spikes'] = xs['sweep'].map(mean_spikes)
        xs[f'Cluster_{cluster}_mean_bline_spikes'] = xs['sweep'].map(mean_bline_spikes)

    if vertical_to_plot:
        import re
        pattern = re.compile(r'^Cluster_[1-3]_mean_spikes$')
        value_vars = [col for col in xs.columns if pattern.match(col)]
        pattern = re.compile(r'^Cluster_[1-4]_mean_bline_spikes$')
        blines = [col for col in xs.columns if pattern.match(col)]
        xs.rename(columns={'control_subtracted': 'Error'}, inplace=True)
        id_vars = ['sweep', 'Pull', 'Condition', 'Error', 'x_decels'] + blines
        xs = xs.melt(id_vars=id_vars, value_vars=value_vars, value_name='Mean Spike Rate', var_name='Cluster')
        return xs
    else:
        return xs
