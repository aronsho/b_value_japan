# The hypothesis H1-H4 are tested with various parameter combinations.
from itertools import product
import pandas as pd
import numpy as np

from seismostats import Catalog
from seismostats.analysis import (
    BPositiveBValueEstimator,
    ClassicBValueEstimator,
    estimate_mc_maxc)
from functions.general_functions import dist_to_ref

# ====== 1: definition of parameters ===========

# magnitude threshold of what is a large event
magnitude_thresholds = [5.5, 6.0, 6.5]

# b-positive estimation method:
# - global: magnitude differences are estimated first for the entire sequence
# corresponding group (near vs. far events). local means that the events are
# first grouped into near and far events, and then the magnitude differences
# are estimated for each group.
b_method = ['global', 'local']

# rupture length relation: wells and coppersmith (1994) provide relations
# for surface and subsurface ruptures.
rupture_length_relation = ['surface', 'subsurface']

# time window after the mainshock that is considered after each event
days_after = [50, 100, 300]

# distance to coast: events are excluded if they are farther away from the
# coast
distance_to_coast = [30, 40, 50]

# dimension: we perform the analysis in 2 dimensions (the sequences are
# defined by a cylinder of twice the rupture length) and 3 dimensions (the
# sequences are defined by a sphere of twice the rupture length)
dimension = [3]

# exclude days of aftershocks
exclude_aftershocks_days = [2]

# ====== 2: find all combinations =============
parameter_combinations = list(product(
    magnitude_thresholds,
    b_method,
    rupture_length_relation,
    days_after,
    distance_to_coast,
    dimension,
    exclude_aftershocks_days,
))

# ====== 2.b: fixed parameters ===========================
correction_factor = 0.2
dmc = 0.1

# ====== 3: get data ===========================
print(len(parameter_combinations), 'parameter combinations found.')
n_combinations = len(parameter_combinations)
for ii in range(0, n_combinations):
    print('Processing combination {} of {}'.format(ii + 1, n_combinations))

    # retrieve the parameters for ii-th combination
    magnitude_threshold = parameter_combinations[ii][0]
    b_method = parameter_combinations[ii][1]
    rupture_length_relation = parameter_combinations[ii][2]
    days_after = parameter_combinations[ii][3]
    days_after = pd.Timedelta(days=days_after)
    distance_to_coast = parameter_combinations[ii][4]
    dimension = parameter_combinations[ii][5]
    exclude_aftershocks_days = parameter_combinations[ii][6]
    exclude_aftershocks_days = pd.Timedelta(days=exclude_aftershocks_days)
    print('- Parameters: ', parameter_combinations[ii])

    # load the catalogs
    print('loading data...')
    filename = 'df_japan_buffered_catalog' + \
        str(distance_to_coast) + 'km_' + str(dimension) + 'D.csv'
    cat_close_buffer = pd.read_csv(
        'data/' + filename, index_col=0)
    cat_close_buffer['time'] = pd.to_datetime(
        cat_close_buffer['time'], format='mixed')
    cat_close_buffer = Catalog(cat_close_buffer)
    cat_close_buffer.delta_m = 0.1
    delta_m = 0.1

    filename = 'df_japan_buffered_catalog400km_' + str(dimension) + 'D.csv'
    cat_400km_buffer = pd.read_csv(
        'data/' + filename, index_col=0)
    cat_400km_buffer['time'] = pd.to_datetime(
        cat_400km_buffer['time'], format='mixed')
    cat_400km_buffer = Catalog(cat_400km_buffer)

    # filter the catalogs
    mc = 0.7
    cat_close_buffer = cat_close_buffer[cat_close_buffer['depth'] <= 150]
    cat_close_buffer = cat_close_buffer[cat_close_buffer.magnitude >= mc]
    cat_close_buffer = cat_close_buffer[cat_close_buffer['time']
                                        >= pd.to_datetime('2000-01-01')]
    cat_400km_buffer = cat_400km_buffer[cat_400km_buffer['depth'] <= 150]
    cat_400km_buffer = cat_400km_buffer[cat_400km_buffer.magnitude >= mc]
    cat_400km_buffer = cat_400km_buffer[cat_400km_buffer['time']
                                        >= pd.to_datetime('2000-01-01')]

    # ====== 4: get the sequences ==================
    df_japan_Large_400km = cat_400km_buffer.copy()
    df_japan_Large_close = cat_close_buffer.copy()

    # first find all the earthquakes that are above the magnitude threshold
    df_japan_Large_400km = df_japan_Large_400km[
        df_japan_Large_400km['magnitude'] >= magnitude_threshold]
    df_japan_Large_400km = df_japan_Large_400km.sort_values('time')

    df_japan_Large_close = df_japan_Large_close[
        df_japan_Large_close['magnitude'] >= magnitude_threshold]
    df_japan_Large_close = df_japan_Large_close.sort_values('time')
    df_japan_Large_close = df_japan_Large_close[
        df_japan_Large_close['time'] > pd.to_datetime(
            '2000-01-01') + days_after]

    # rupture length
    if rupture_length_relation == 'surface':
        a = -3.22
        b = 0.69
    elif rupture_length_relation == 'subsurface':
        a = -2.44
        b = 0.59
    rupture_length = 10**(a + b * df_japan_Large_400km['magnitude'].values)
    df_japan_Large_400km.loc[:, 'rupture_length'] = rupture_length
    rupture_length = 10**(a + b * df_japan_Large_close['magnitude'].values)

    df_japan_Large_close.loc[:, 'rupture_length'] = rupture_length

    # go through all large events
    radius_close = 0.5  # radius as a fraction of the rupture length
    radius_far = 2  # radius as a fraction of the rupture length
    days_before = pd.Timedelta(days=10*365)
    idx_drop = []
    large_sequences = []
    sequence_main_idx = []
    print('finding sequences...')
    print(len(df_japan_Large_close), 'large events found in the close catalog')
    for kk in df_japan_Large_close.index:
        # define the main event
        main_event = df_japan_Large_close.loc[kk]

        # define sequence
        R = main_event['rupture_length']
        sequence_start = main_event['time'] - days_before
        sequence_stop = main_event['time'] + days_after

        # make the start and end of the sequence such that there is no overlap
        # with other M>6 events
        x_ref = main_event['x']
        y_ref = main_event['y']
        z_ref = main_event['z']
        for jj in df_japan_Large_400km.index:
            if kk == jj:
                continue
            # estimate the distance
            x = df_japan_Large_400km.loc[jj, 'x']
            y = df_japan_Large_400km.loc[jj, 'y']
            z = df_japan_Large_400km.loc[jj, 'z']
            if dimension == 3:
                distance = dist_to_ref(x, x_ref, y, y_ref, z, z_ref)
            elif dimension == 2:
                distance = dist_to_ref(x, x_ref, y, y_ref)

            if distance < radius_far * (
                    R + df_japan_Large_400km.loc[jj, 'rupture_length']):
                # if the other M>6 event is after the main event, then we
                # only consider events up until that moment
                if df_japan_Large_400km.loc[jj, 'time'] > main_event['time']:
                    sequence_stop = min(
                        sequence_stop, df_japan_Large_400km.loc[jj, 'time'])
                # if the other M>6 event is before the main event, then we
                # only consider events after that moment + days_after
                elif df_japan_Large_400km.loc[jj, 'time'] < main_event['time']:
                    sequence_start = max(
                        sequence_start, df_japan_Large_400km.loc[
                            jj, 'time'] + days_after)

        # drop event if the start is after the main event
        if sequence_start > main_event['time']:
            idx_drop.append(kk)
            continue

        # find all events within twice the rupture length of the main event
        # (within the 40 km buffer)
        x = cat_close_buffer['x']
        y = cat_close_buffer['y']
        z = cat_close_buffer['z']
        if dimension == 3:
            distances = dist_to_ref(x, x_ref, y, y_ref, z, z_ref)
        elif dimension == 2:
            distances = dist_to_ref(x, x_ref, y, y_ref)
        idx_dist = distances <= radius_far * R

        # define the sequence
        idx_sequences = idx_dist & (
            cat_close_buffer['time'] > sequence_start) & (
            cat_close_buffer['time'] < sequence_stop)
        sequence = cat_close_buffer[idx_sequences].copy()

        # mark the distance to the main event
        sequence['distance_to_main'] = distances[idx_sequences]

        # delete main event and exclude_aftershocks_days of aftershocks
        idx_beforemain = sequence['time'] < main_event['time']
        idx_aftermain = (
            sequence['time'] > main_event['time'] + exclude_aftershocks_days)
        sequence = pd.concat([
            sequence[idx_beforemain],
            sequence[idx_aftermain]
        ])
        sequence = sequence.sort_values('time')

        # only consider sequences with more than 200 events
        if len(sequence) <= 200:
            idx_drop.append(kk)
            continue

        # add sequence to the list of sequences
        large_sequences.append(sequence)
        sequence_main_idx.append(kk)

    df_japan_Large_close = df_japan_Large_close.drop(idx_drop)
    print(len(large_sequences), 'sequences found.')

    # ====== 5: estimate b-values before, after, far, close ======
    print('estimating b-values...')

    # overall features
    b_sequence = []
    std_sequence = []
    p_l_sequence = []

    # define lists for the b-values
    b_close_after = []
    b_close_before = []
    b_far_after = []
    b_far_before = []
    std_close_after, p_l_close_after = [], []
    std_close_before, p_l_close_before = [], []
    std_far_after, p_l_far_after = [], []
    std_far_before, p_l_far_before = [], []

    b_before = []
    b_after = []
    b_close = []
    b_far = []
    std_before, p_l_before = [], []
    std_after, p_l_after = [], []
    std_close, p_l_close = [], []
    std_far, p_l_far = [], []

    b_before1 = []
    b_before2 = []
    std_before1, p_l_before1 = [], []
    std_before2, p_l_before2 = [], []

    b_before1_close = []
    b_before2_close = []
    std_before1_close, p_l_before1_close = [], []
    std_before2_close, p_l_before2_close = [], []

    n_check = 50 if b_method == 'global' else 100
    estimator = BPositiveBValueEstimator()
    for kk, sequence in enumerate(large_sequences):
        # exclude main event from the sequence
        main_event = df_japan_Large_close.loc[sequence_main_idx[kk]]

        # estimate mc of the sequence
        mc, _ = estimate_mc_maxc(
            sequence.magnitude,
            fmd_bin=delta_m,
            correction_factor=correction_factor)

        # estimate differences
        estimator.calculate(sequence.magnitude, mc=mc,
                            delta_m=delta_m, times=sequence.time, dmc=dmc)
        if b_method == 'global':
            mags = estimator.magnitudes
            times = estimator.times
            distances = sequence['distance_to_main'].values[estimator.idx]
            estimator_2 = ClassicBValueEstimator()
        elif b_method == 'local':
            mags = sequence['magnitude'].values
            idx = mags >= mc
            mags = mags[idx]
            times = sequence['time'].values[idx]
            distances = sequence['distance_to_main'].values[idx]

            estimator_2 = BPositiveBValueEstimator()

        # calculate the b-value for the entire sequence
        b_sequence.append(estimator.b_value)
        std_sequence.append(estimator.std)
        p_l_sequence.append(estimator.p_lilliefors())

        # estimate indexes
        idx_close = distances <= radius_close * \
            main_event.rupture_length
        idx_before = times < main_event['time']
        idx_after = times > main_event['time']

        # estimate magnitudes and times for the different groups
        mags_close_after = mags[idx_close & idx_after]
        times_close_after = times[idx_close & idx_after]
        mags_close_before = mags[idx_close & idx_before]
        times_close_before = times[idx_close & idx_before]
        mags_far_after = mags[~idx_close & idx_after]
        times_far_after = times[~idx_close & idx_after]
        mags_far_before = mags[~idx_close & idx_before]
        times_far_before = times[~idx_close & idx_before]

        mags_close = mags[idx_close]
        times_close = times[idx_close]
        mags_far = mags[~idx_close]
        times_far = times[~idx_close]
        mags_before = mags[idx_before]
        times_before = times[idx_before]
        mags_after = mags[idx_after]
        times_after = times[idx_after]

        # close, after
        if len(mags_close_after) > n_check:
            if b_method == 'global':
                estimator_2.calculate(
                    mags_close_after, mc=dmc, delta_m=delta_m)
            elif b_method == 'local':
                estimator_2.calculate(
                    mags_close_after, mc=mc, delta_m=delta_m, dmc=dmc,
                    times=times_close_after)
            b_close_after.append(estimator_2.b_value)
            std_close_after.append(estimator_2.std)
            p_l_close_after.append(estimator_2.p_lilliefors())
        else:
            b_close_after.append(np.nan)
            std_close_after.append(np.nan)
            p_l_close_after.append(np.nan)

        # close, before
        if len(mags_close_before) > n_check:
            if b_method == 'global':
                estimator_2.calculate(
                    mags_close_before, mc=dmc, delta_m=delta_m)
            elif b_method == 'local':
                estimator_2.calculate(
                    mags_close_before, mc=mc, delta_m=delta_m, dmc=dmc,
                    times=times_close_before)
            b_close_before.append(estimator_2.b_value)
            std_close_before.append(estimator_2.std)
            p_l_close_before.append(estimator_2.p_lilliefors())
        else:
            b_close_before.append(np.nan)
            std_close_before.append(np.nan)
            p_l_close_before.append(np.nan)

        # far, after
        if len(mags_far_after) > n_check:
            if b_method == 'global':
                estimator_2.calculate(mags_far_after, mc=dmc, delta_m=delta_m)
            elif b_method == 'local':
                estimator_2.calculate(
                    mags_far_after, mc=mc, delta_m=delta_m, dmc=dmc,
                    times=times_far_after)
            b_far_after.append(estimator_2.b_value)
            std_far_after.append(estimator_2.std)
            p_l_far_after.append(estimator_2.p_lilliefors())
        else:
            b_far_after.append(np.nan)
            std_far_after.append(np.nan)
            p_l_far_after.append(np.nan)

        # far, before
        if len(mags_far_before) > n_check:
            if b_method == 'global':
                estimator_2.calculate(mags_far_before, mc=dmc, delta_m=delta_m)
            elif b_method == 'local':
                estimator_2.calculate(
                    mags_far_before, mc=mc, delta_m=delta_m, dmc=dmc,
                    times=times_far_before)
            b_far_before.append(estimator_2.b_value)
            std_far_before.append(estimator_2.std)
            p_l_far_before.append(estimator_2.p_lilliefors())
        else:
            b_far_before.append(np.nan)
            std_far_before.append(np.nan)
            p_l_far_before.append(np.nan)

        # before
        if len(mags_before) > n_check:
            if b_method == 'global':
                estimator_2.calculate(mags_before, mc=dmc, delta_m=delta_m)
            elif b_method == 'local':
                estimator_2.calculate(
                    mags_before, mc=mc, delta_m=delta_m, dmc=dmc,
                    times=times_before)
            b_before.append(estimator_2.b_value)
            std_before.append(estimator_2.std)
            p_l_before.append(estimator_2.p_lilliefors())
        else:
            b_before.append(np.nan)
            std_before.append(np.nan)
            p_l_before.append(np.nan)

        # before 1 and before 2
        len_half = int(len(mags_before)/2)
        mags_before1 = mags_before[:len_half]
        times_before1 = times_before[:len_half]
        mags_before2 = mags_before[len_half:]
        times_before2 = times_before[len_half:]

        if len(mags_before1) >= n_check:
            if b_method == 'global':
                estimator_2.calculate(mags_before1, mc=dmc, delta_m=delta_m)
            elif b_method == 'local':
                estimator_2.calculate(
                    mags_before1, mc=mc, delta_m=delta_m, dmc=dmc,
                    times=times_before1)
            b_before1.append(estimator_2.b_value)
            std_before1.append(estimator_2.std)
            p_l_before1.append(estimator_2.p_lilliefors())
        else:
            b_before1.append(np.nan)
            std_before1.append(np.nan)
            p_l_before1.append(np.nan)
        if len(mags_before2) >= n_check:
            if b_method == 'global':
                estimator_2.calculate(mags_before2, mc=dmc, delta_m=delta_m)
            elif b_method == 'local':
                estimator_2.calculate(
                    mags_before2, mc=mc, delta_m=delta_m, dmc=dmc,
                    times=times_before2)
            b_before2.append(estimator_2.b_value)
            std_before2.append(estimator_2.std)
            p_l_before2.append(estimator_2.p_lilliefors())
        else:
            b_before2.append(np.nan)
            std_before2.append(np.nan)
            p_l_before2.append(np.nan)

        # before_close 1 and before_close 2
        len_half = int(len(mags_close_before)/2)
        mags_before1_close = mags_close_before[:len_half]
        times_before1_close = times_close_before[:len_half]
        mags_before2_close = mags_close_before[len_half:]
        times_before2_close = times_close_before[len_half:]

        if len(mags_before1_close) >= n_check:
            if b_method == 'global':
                estimator_2.calculate(
                    mags_before1_close, mc=dmc, delta_m=delta_m)
            elif b_method == 'local':
                estimator_2.calculate(
                    mags_before1_close, mc=mc, delta_m=delta_m, dmc=dmc,
                    times=times_before1_close)
            b_before1_close.append(estimator_2.b_value)
            std_before1_close.append(estimator_2.std)
            p_l_before1_close.append(estimator_2.p_lilliefors())
        else:
            b_before1_close.append(np.nan)
            std_before1_close.append(np.nan)
            p_l_before1_close.append(np.nan)
        if len(mags_before2_close) >= n_check:
            if b_method == 'global':
                estimator_2.calculate(
                    mags_before2_close, mc=dmc, delta_m=delta_m)
            elif b_method == 'local':
                estimator_2.calculate(
                    mags_before2_close, mc=mc, delta_m=delta_m, dmc=dmc,
                    times=times_before2_close)
            b_before2_close.append(estimator_2.b_value)
            std_before2_close.append(estimator_2.std)
            p_l_before2_close.append(estimator_2.p_lilliefors())
        else:
            b_before2_close.append(np.nan)
            std_before2_close.append(np.nan)
            p_l_before2_close.append(np.nan)

        # after
        if len(mags_after) >= n_check:
            if b_method == 'global':
                estimator_2.calculate(mags_after, mc=dmc, delta_m=delta_m)
            elif b_method == 'local':
                estimator_2.calculate(
                    mags_after, mc=mc, delta_m=delta_m, dmc=dmc,
                    times=times_after)
            b_after.append(estimator_2.b_value)
            std_after.append(estimator_2.std)
            p_l_after.append(estimator_2.p_lilliefors())
        else:
            b_after.append(np.nan)
            std_after.append(np.nan)
            p_l_after.append(np.nan)

        # close
        if len(mags_close) >= n_check:
            if b_method == 'global':
                estimator_2.calculate(mags_close, mc=dmc, delta_m=delta_m)
            elif b_method == 'local':
                estimator_2.calculate(
                    mags_close, mc=mc, delta_m=delta_m, dmc=dmc,
                    times=times_close)
            b_close.append(estimator_2.b_value)
            std_close.append(estimator_2.std)
            p_l_close.append(estimator_2.p_lilliefors())
        else:
            b_close.append(np.nan)
            std_close.append(np.nan)
            p_l_close.append(np.nan)

        # far
        if len(mags_far) >= n_check:
            if b_method == 'global':
                estimator_2.calculate(mags_far, mc=dmc, delta_m=delta_m)
            elif b_method == 'local':
                estimator_2.calculate(
                    mags_far, mc=mc, delta_m=delta_m, dmc=dmc, times=times_far)
            b_far.append(estimator_2.b_value)
            std_far.append(estimator_2.std)
            p_l_far.append(estimator_2.p_lilliefors())
        else:
            b_far.append(np.nan)
            std_far.append(np.nan)
            p_l_far.append(np.nan)

    b_sequence = np.array(b_sequence)
    std_sequence = np.array(std_sequence)
    p_l_sequence = np.array(p_l_sequence)

    b_close_after = np.array(b_close_after)
    b_close_before = np.array(b_close_before)
    b_far_after = np.array(b_far_after)
    b_far_before = np.array(b_far_before)
    std_close_after = np.array(std_close_after)
    std_close_before = np.array(std_close_before)
    std_far_after = np.array(std_far_after)
    std_far_before = np.array(std_far_before)
    p_l_close_after = np.array(p_l_close_after)
    p_l_close_before = np.array(p_l_close_before)
    p_l_far_after = np.array(p_l_far_after)
    p_l_far_before = np.array(p_l_far_before)

    b_before = np.array(b_before)
    b_after = np.array(b_after)
    b_close = np.array(b_close)
    b_far = np.array(b_far)
    std_before = np.array(std_before)
    std_after = np.array(std_after)
    std_close = np.array(std_close)
    std_far = np.array(std_far)
    p_l_before = np.array(p_l_before)
    p_l_after = np.array(p_l_after)
    p_l_close = np.array(p_l_close)
    p_l_far = np.array(p_l_far)

    b_before1 = np.array(b_before1)
    b_before2 = np.array(b_before2)
    std_before1 = np.array(std_before1)
    std_before2 = np.array(std_before2)
    p_l_before1 = np.array(p_l_before1)
    p_l_before2 = np.array(p_l_before2)

    b_before1_close = np.array(b_before1_close)
    b_before2_close = np.array(b_before2_close)
    std_before1_close = np.array(std_before1_close)
    std_before2_close = np.array(std_before2_close)
    p_l_before1_close = np.array(p_l_before1_close)
    p_l_before2_close = np.array(p_l_before2_close)

    # ===== save the results ========================
    # save all the b-values in a dataframe
    df_b_values = pd.DataFrame({
        'b_sequence': b_sequence,
        'std_sequence': std_sequence,
        'p_l_sequence': p_l_sequence,
        'b_close_after': b_close_after,
        'std_close_after': std_close_after,
        'p_l_close_after': p_l_close_after,
        'b_close_before': b_close_before,
        'std_close_before': std_close_before,
        'p_l_close_before': p_l_close_before,
        'b_far_after': b_far_after,
        'std_far_after': std_far_after,
        'p_l_far_after': p_l_far_after,
        'b_far_before': b_far_before,
        'std_far_before': std_far_before,
        'p_l_far_before': p_l_far_before,
        'b_before': b_before,
        'std_before': std_before,
        'p_l_before': p_l_before,
        'b_after': b_after,
        'std_after': std_after,
        'p_l_after': p_l_after,
        'b_close': b_close,
        'std_close': std_close,
        'p_l_close': p_l_close,
        'b_far': b_far,
        'std_far': std_far,
        'p_l_far': p_l_far,
        'b_before1': b_before1,
        'std_before1': std_before1,
        'p_l_before1': p_l_before1,
        'b_before2': b_before2,
        'std_before2': std_before2,
        'p_l_before2': p_l_before2,
        'b_before1_close': b_before1_close,
        'std_before1_close': std_before1_close,
        'p_l_before1_close': p_l_before1_close,
        'b_before2_close': b_before2_close,
        'std_before2_close': std_before2_close,
        'p_l_before2_close': p_l_before2_close
    }, index=sequence_main_idx)

    save_filename = 'results/results_parameter_variation/df_b_values_' + \
        str(magnitude_threshold) + 'M_' + b_method + '_' + \
        rupture_length_relation + '_' + str(days_after.days) + 'days_' + \
        str(distance_to_coast) + 'km_' + str(dimension) + 'D_' + \
        str(exclude_aftershocks_days.days) + 'days' + \
        'X.csv'
    df_b_values.to_csv(save_filename)
