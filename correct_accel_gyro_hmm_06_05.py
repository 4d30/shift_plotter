#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 22:50:30 2024

@author: dylanrichards
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
import scipy.interpolate
from scipy.stats import lognorm, norm
import argparse
import glob
from pathlib import Path
import os
from sklearn.neighbors import KernelDensity
import pickle

def plot_shift(avg_prob,path,shifts,convert_names):
    fig, axs = plt.subplots(6, 2, figsize=(15, 10))
    gs = fig.add_gridspec(6, 2, width_ratios=[1, 1])

    # Creating subplots in the left column
    ax_left = [fig.add_subplot(gs[i, 0]) for i in range(6)]

    # Creating subplot in the right column that spans all rows used by the left column
    ax_right = fig.add_subplot(gs[0:6, 1])

    # Plotting the time series in the left column
    for i, (ax, shift) in enumerate(zip(ax_left, shifts)):
        ax.plot(np.arange(len(shift)) / 215, shift)
        if i != (len(shifts)-1):
            ax.set_xticklabels([])
        ax.set_ylim([-0.1, 1.5])
        ax.axhline(1, color='r', linestyle='--')
        ax.axhline(0, color='r', linestyle='--')
        ax.set_title(', '.join(convert_names[i]),fontsize=10)

    # Plotting the heatmap and paths in the right column
    ax_right.imshow(avg_prob.T, aspect='auto', interpolation='none')
    ax_right.plot(np.arange(len(path)), path, color='red', marker='o', markersize=5, alpha=0.5, linestyle='-', linewidth=2, label='Viterbi Path 1')


    # ax_right.set_yticklabels('')


    return fig,axs

def estimate_likelihood_from_data(accel_df, accel_fs=215):
    accel_cols = ['Accl X(g)', 'Accl Y(g)', 'Accl Z(g)']
    gyro_cols = ['Velocity X(dps)', 'Velocity Y(dps)', 'Velocity Z(dps)']

    accel_norm = np.linalg.norm(accel_df.loc[:,accel_cols],axis=1)
    gyro_norm = np.linalg.norm(accel_df.loc[:,gyro_cols],axis=1) / 1000

    combined = np.column_stack([accel_norm,gyro_norm])
    b,a = scipy.signal.butter(3, 0.3, btype='low', fs=accel_fs)
    filtered = scipy.signal.filtfilt(b,a,combined,axis=0)

    gyro_params = lognorm.fit(filtered[:,1])
    accel_params = norm.fit(filtered[:,0])
    return accel_params, gyro_params
class JointProbability():

    def __init__(self, accel_df=None):
        if accel_df is None:
            accel_params = (1, 0.1609509350680874)
            gyro_params = (0.807021431359285,-0.01783034260529351,0.04829919821164311)
        else:
            accel_params, gyro_params = estimate_likelihood_from_data(accel_df)
        self.accel_params = accel_params
        self.gyro_params = gyro_params


    def predict(self, accel_norm, gyro_norm):
        accel_likelihood = norm.pdf(accel_norm, loc=self.accel_params[0],
                                    scale=self.accel_params[1])
        gyro_likelihood = lognorm.pdf(gyro_norm, s=self.gyro_params[0],
                                      loc=self.gyro_params[1],
                                      scale=self.gyro_params[2])
        return accel_likelihood * gyro_likelihood

def find_low_motion_regions(motion_signal, min_motion_sample=208*5, motion_threshold=1e-5):
    '''
    Returns inclusive indices!
    '''
    # motion_threshold = 1e-5
    low_motion_ind = motion_signal < motion_threshold

    diff = np.diff(low_motion_ind.astype(int))
    starts = np.where(diff == 1)[0] + 1  # start indexes of low motion
    ends = np.where(diff == -1)[0]
    if low_motion_ind[0]:
        starts = np.insert(starts, 0, 0)

    # If the signal ends with a low motion, add the last index to ends
    if low_motion_ind[-1]:
        ends = np.append(ends, len(low_motion_ind) - 1)
    low_motion_periods = list(zip(starts, ends))
    filtered = []
    for start,stop in low_motion_periods:
        duration = stop - start + 1
        if duration > min_motion_sample:
            filtered.append([start, stop])
    return filtered

def compute_shift_probs(accel_df, joint_dist, low_motion_adjust=True):
    '''
    Computes the probability of each sample occuring with each of the possible
    frame shifts

    joint_dist is JointProbability object

    '''
    accel_cols = ['Accl X(g)', 'Accl Y(g)', 'Accl Z(g)']
    gyro_cols = ['Velocity X(dps)', 'Velocity Y(dps)', 'Velocity Z(dps)']
    unconvert_data = accel_df.copy()
    unconvert_data.loc[:,accel_cols] = unconvert_data.loc[:,accel_cols] / 0.244
    unconvert_data.loc[:,gyro_cols] = unconvert_data.loc[:,gyro_cols] / 35

    stack_data = unconvert_data.loc[:,accel_cols + gyro_cols].values
    accel_fs = int(np.median(1/np.diff(accel_df['Time(ms)'].values/1000)))
    # low pass the accelerometer data for a more stable reading, overcomes
    # most motion
    b,a = scipy.signal.butter(3, 0.3, btype='low', fs=accel_fs)
    filtered = scipy.signal.filtfilt(b,a,stack_data,axis=0)


    norm_data = []
    shifts = []
    convert_names = []
    shift_flat_dist = []
    motion = []

    for shift_amount in range(6):
        # break
        rolled_names = np.roll(['AX','AY','AZ','GX','GY','GZ'],shift=shift_amount)
        convert = [f'{r}->{f}' for r,f in zip(rolled_names,['AX','AY','AZ','GX','GY','GZ'])]
        rolled_data = np.roll(filtered, shift=shift_amount, axis=1)
        convert_names.append(convert)
        assume_accel = rolled_data[:,0:3] * 0.244 # Rescale to accelerometer
        assume_gyro = rolled_data[:,3:] * 35      # Rescale to gyroscope

        assume_accel_norm = np.linalg.norm(assume_accel, axis=1)
        assume_gyro_norm = np.linalg.norm(assume_gyro, axis=1) / 1000
        # compute a motion metric for each possible accelerometer mixture
        # we'll use this later to look for areas where the swapping is being
        # confused when a gyro and accel value (both close to 0) are being confused
        accel_motion = np.hstack([0,np.linalg.norm(np.diff(assume_accel,axis=0),axis=1)])
        motion.append(accel_motion)

        likelihood = joint_dist.predict(assume_accel_norm, assume_gyro_norm)

        # find regions where the presumed z-axis is laying flat (==1, or ==-1)
        # plus_z_dot = np.einsum('ij,j->i',assume_accel,np.array([0,0,1]))
        # minux_z_dot = np.einsum('ij,j->i',assume_accel,np.array([0,0,-1]))
        # dist_to_flat = np.clip(np.abs(assume_accel[:,2]), 0, 1)
        dist_to_flat = np.abs(assume_accel[:,2])

        shift_flat_dist.append(dist_to_flat)

        norm_data.append(np.column_stack([assume_accel_norm, assume_gyro_norm]))

        shifts.append(likelihood)


    # This code block will identify regions where the following occurs:
    # 1. There is a block of very low motion for at least 5 seconds
    # 2. There is no clear accel/gyro orientation. This can occur when the accelerometer
    #   and gyro values in one axis is 0 (basically sitting on a table), which
    # results in 2 - 3 equally likely states.
    # When 1 and 2 happen together, just set the probability of each shift state
    # equal to the null probability of 1/6
    #

    raw_likelihood = np.column_stack(shifts)
    motion = np.min(np.column_stack(motion), axis=1)
    low_motion_inds = find_low_motion_regions(motion, min_motion_sample=accel_fs*5,
                                              motion_threshold=1e-6)
    low_motion_bool = np.zeros([len(motion),],dtype=bool)
    for start, stop in low_motion_inds:
        low_motion_bool[start:stop+1] = True

    probabilities = raw_likelihood / np.sum(raw_likelihood,axis=1,keepdims=True)


    # find placed where there are 3 or more low probability axes and 2 or more high probability
    # axes. This removes situations where there is a single high probability state
    tied_probs = (np.sum(probabilities<0.1,axis=1)>=3) & (np.sum(probabilities>0.2,axis=1)>=2)


    # combine with low motion
    tied_probs_and_low_motion = tied_probs & low_motion_bool
    # multiply by the abs z-axis distance for each possible shift.
    if low_motion_adjust:
        # instead of trying to adjust probabilities, set the probabilities even
        # and rely on viterbi to carry
        raw_likelihood[tied_probs_and_low_motion] = 1/6

        # renormalize
        probabilities = raw_likelihood / np.sum(raw_likelihood,axis=1,keepdims=True)


    return probabilities, convert_names, norm_data, raw_likelihood

def viterbi(emissions, trans_probs, init_probs):
    num_steps = emissions.shape[0]
    num_states = emissions.shape[1]

    # Path probabilities at each step
    path_probs = np.zeros((num_steps, num_states))
    # Path back pointers
    paths = np.zeros((num_steps, num_states), dtype=int)

    # Initialize with the first emission and initial state probabilities
    path_probs[0, :] = np.log(emissions[0] * init_probs + 1e-6)

    # Fill in the Viterbi matrix
    for t in range(1, num_steps):
        for s in range(num_states):
            # Calculate the probability of each state leading to this state
            prob = path_probs[t - 1] + np.log(trans_probs[:, s]+1e-6) + np.log(emissions[t, s]+1e-6)
            # Find the best previous state
            best_prev_state = np.argmax(prob)
            path_probs[t, s] = prob[best_prev_state]
            paths[t, s] = best_prev_state

    # Backtrack to find the most probable path
    best_path = np.zeros(num_steps, dtype=int)
    best_path[-1] = np.argmax(path_probs[-1, :])
    for t in range(num_steps - 2, -1, -1):
        best_path[t] = paths[t + 1, best_path[t + 1]]

    return best_path,path_probs


def increase_smoothness(transition_matrix, transition_modifier):
    '''
    reduce the off diagonal elements of the transition matrix by multipling by
    transintion_modifier, then re-normalize. The end effect is reducing the
    willingness of the state to change from the current one.
    '''
    other_transitions = transition_matrix * (1-np.eye(len(transition_matrix)))
    new_mat_unbalanced = (transition_matrix * np.eye(len(transition_matrix))) + other_transitions * transition_modifier
    new_trans_mat = new_mat_unbalanced / np.sum(new_mat_unbalanced,axis=1,keepdims=True)
    return new_trans_mat

def correct_frame_shift(accel_df, shift_state, accel_fs=215, seconds_remove=2):
    '''
    Correct frame shift by upsampling the shift_state sequence by the
    expected once per second downsampling

    Then unconvert the original accelerometer data, roll through each state
    and save the selection in each state that was found to be the best

    seconds_remove = number of seconds (forward seconds and backward seconds)
        to set to nan around state transition

    '''
    shift_ind = np.arange(len(shift_state))*accel_fs
    shift_interp = scipy.interpolate.interp1d(shift_ind, shift_state,kind='zero',
                                              bounds_error=False,fill_value='extrapolate')
    upsample_shift_state = shift_interp(np.arange(len(accel_df)))

    accel_cols = ['Accl X(g)', 'Accl Y(g)', 'Accl Z(g)']
    gyro_cols = ['Velocity X(dps)', 'Velocity Y(dps)', 'Velocity Z(dps)']
    unconvert_data = accel_df.copy()
    unconvert_data.loc[:,accel_cols] = unconvert_data.loc[:,accel_cols] / 0.244
    unconvert_data.loc[:,gyro_cols] = unconvert_data.loc[:,gyro_cols] / 35

    stack_data = unconvert_data.loc[:,accel_cols + gyro_cols].values
    adjusted_data = np.empty(stack_data.shape,dtype=np.float64)
    adjusted_data[:] = np.nan

    convert_names = []
    for shift_amount in range(6):
        rolled_names = np.roll(['AX','AY','AZ','GX','GY','GZ'],shift=shift_amount)
        convert = [f'{r}->{f}' for r,f in zip(rolled_names,['AX','AY','AZ','GX','GY','GZ'])]
        rolled_data = np.roll(stack_data, shift=shift_amount, axis=1)
        convert_names.append(convert)
        assume_accel = rolled_data[:,0:3] * 0.244 # Rescale to accelerometer
        assume_gyro = rolled_data[:,3:] * 35      # Rescale to gyroscope
        combined_signals = np.column_stack([assume_accel, assume_gyro])
        # select the points at the particular shift amount to save
        swap_inds = upsample_shift_state==shift_amount
        adjusted_data[swap_inds,:] = combined_signals[swap_inds,:]

    # find state shift indexes, then set regions around to nan
    # where is current state not equal to next state
    shift_inds = np.where(upsample_shift_state[0:-1] != upsample_shift_state[1:])[0]
    remove_samples = int(seconds_remove * accel_fs)
    # ensure we're within indexing bounds
    left_nan_start  = np.clip(shift_inds - remove_samples,0,len(adjusted_data)-1)
    right_nan_end = np.clip(shift_inds + remove_samples,0,len(adjusted_data)-1)

    removed_regions = np.zeros([len(adjusted_data),])
    for left_ind,right_ind in zip(left_nan_start,right_nan_end):
        adjusted_data[left_ind:right_ind] = np.nan
        removed_regions[left_ind:right_ind] = 1

    new_accel_df = pd.DataFrame(adjusted_data,columns=accel_cols+gyro_cols,
                                index=accel_df.index)
    new_accel_df['Time(ms)'] = accel_df['Time(ms)']
    return new_accel_df, upsample_shift_state, removed_regions


def boxcar_average(data, window_size=215):

    # Number of full windows that fit into the array
    num_full_windows = data.shape[0] // window_size

    # Reshape the data to ignore the remainder elements
    reshaped_data = data[:num_full_windows * window_size].reshape(num_full_windows, window_size, data.shape[1])

    # Calculate the mean across the windows (axis 1)
    window_means = reshaped_data.mean(axis=1)
    return window_means

def create_hmm_params():
    # created by analyzing the transition counts of the data when not
    # modeled as a hmm
    initial_trans_mat_count = [
        [28464, 490, 100, 9, 157, 1068],
        [483, 453, 21, 27, 10, 70],
        [95, 18, 437, 111, 29, 8],
        [4, 25, 111, 4642, 242, 35],
        [174, 8, 22, 246, 3220, 177],
        [1065, 70, 6, 26, 191, 12252]
    ]

    # Convert to a numpy array
    initial_trans_mat_count = np.array(initial_trans_mat_count)

    initial_trans_mat_prob = initial_trans_mat_count / np.sum(initial_trans_mat_count,axis=1,keepdims=True)
    initial_starting_probabilities = initial_trans_mat_count.sum(axis=0) / initial_trans_mat_count.sum()
    return initial_trans_mat_prob, initial_starting_probabilities

def fix_imu(accel_df, plot=False, seconds_remove=2, low_motion_adjust=True):
    '''
    Switches mismatches accelerometer and gyroscope data. Whenever state change
    occurred (accel was called gyro and or vice versa), the corrected accel and
    gyro scope data will have left and right `seconds_remove` set to nan.

    Parameters:
    ----------
        accel_df: pandas dataframe, expects to have the following columns:
            ['Time(ms)','Accl X(g)', 'Accl Y(g)', 'Accl Z(g)',
             'Velocity X(dps)', 'Velocity Y(dps)', 'Velocity Z(dps)']
            as produced by the Sibel shrd processor during the shrd->csv
            proess.
        plot: default=False. Plots the algorithm steps for inspection.
        seconds_remove = number of seconds (forward seconds and backward seconds)
            to set to nan around state transition. IE, if seconds_remove is 2,
            the 4 seconds total is set to nan.
    Returns:
    -------
        accel_df_out: pandas dataframe of corrected accelerometer and gyroscope
            data. Same shape and columns as input accel_df
        upsample_shift_state: Index of expected frame shift
        probabilities: raw state probabilities before viterbi algorithm
        removed_regions: number of seconds to remove (set to nan)

    '''
    columns = accel_df.columns # to restore order later
    accel_fs = int(np.median(1/np.diff(accel_df['Time(ms)'].values/1000)))
    initial_trans_mat_prob, initial_starting_probabilities = create_hmm_params()
    smooth_trans_mat = increase_smoothness(initial_trans_mat_prob,0.01)
    overall_joint_prob = JointProbability()

    probabilities, convert_names, shifts, raw_likelihood = compute_shift_probs(accel_df,
                                                                               overall_joint_prob,
                                                                               low_motion_adjust=low_motion_adjust)


    # modify the probabilities (which are just the normalized likelihood values)
    # such that in the situation where there is no good shift candidate,
    # we linearly interpolate the probabilities with a uniform probability distribution
    # with the end result being that in situations where the overall
    # likelihood is low, a state change is unlikely, and the current state will
    # remain.
    # max raw likelihood seems to be about 35.1338. Normalize and then clip to keep
    # between 0 and 1.
    max_likelihood = 35.1338
    max_raw_likelihood = np.clip(np.max(raw_likelihood,axis=1,keepdims=True) / max_likelihood, 0, 1)

    num_states = 6
    adj_prob = ((probabilities * max_raw_likelihood)
                + (np.ones_like(probabilities) / num_states) * (1-max_raw_likelihood))

    # average to every second for faster computation
    avg_prob = boxcar_average(adj_prob, window_size=accel_fs)

    path,path_probs = viterbi(avg_prob, smooth_trans_mat, initial_starting_probabilities)

    adjusted, upsample_shift_state, removed_regions = correct_frame_shift(accel_df,
                                                                          path,
                                                                          accel_fs=accel_fs,
                                                                          seconds_remove=seconds_remove)
    if plot:
        f,axs = plot_shift(avg_prob,path,shifts,convert_names)

    adjusted = adjusted[columns]
    return adjusted, upsample_shift_state, adj_prob, removed_regions

def main(input_csv, output_csv):
    # Load the CSV file
    accel_df = pd.read_csv(input_csv)
    accel_df_out, shift_state_index, raw_probabilities,removed_regions = fix_imu(accel_df,
                                                                                 plot=False,
                                                                                 seconds_remove=2,
                                                                                 low_motion_adjust=True)
    # Perform the operation
    # Save the modified DataFrame to the specified output CSV file
    accel_df_out.to_csv(output_csv, index=False,
                        na_rep='nan', compression='gzip',
                        float_format='%.6f')

#    print(f"Output saved to {output_csv}")

if __name__ == '__main__':
    # Create the parser


    parser = argparse.ArgumentParser(description="Process accelerometer and gyroscope data.")

    # Add arguments for input and output CSV files
    parser.add_argument('-i', '--input', required=True, help="Input CSV filename")
    parser.add_argument('-o', '--output', required=True, help="Output CSV filename")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the processed arguments
    main(args.input, args.output)



