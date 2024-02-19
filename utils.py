import pandas as pd
import os, random
import numpy as np
from itertools import chain
from scipy.signal import stft
import tensorflow as tf


train_acceleration_data_path = '/home/lasii/Research/dataset/HAR/harAGE/Train/'
deploy_acceleration_data_path = '/home/lasii/Research/dataset/HAR/harAGE/Devel/'

train_heart_data_path = '/home/lasii/Research/dataset/HAR/harAGE/Train/'
deploy_heart_data_path = '/home/lasii/Research/dataset/HAR/harAGE/Devel/'

train_label_path = '/home/lasii/Research/dataset/HAR/harAGE/Train_metadata.csv'
deploy_label_path = '/home/lasii/Research/dataset/HAR/harAGE/Devel_metadata.csv'


def read_heart(dir):
    heart_rate = []
    for filename in sorted(os.listdir(dir)):
        if filename.endswith(".csv"):
            print('Reading File:= ', filename)
            df = pd.read_csv(os.path.join(dir, filename), sep = ';').iloc[:,:3]
            heart_rate.append(df)
        else:
            continue
    return np.array(heart_rate)


def read_acceleration(dir):
    acceleration = []
    for filename in sorted(os.listdir(dir)):
        if filename.endswith(".csv"): 
            print('Reading File:= ', filename)
            df = pd.read_csv(os.path.join(dir, filename), sep = ';').iloc[:,-3:]
            x = list(map(int, list(chain.from_iterable(df['accelerometer_milliG_xAxis'].apply(lambda x: x.strip('[').strip(']').split(',')).values))))
            y = list(map(int, list(chain.from_iterable(df['accelerometer_milliG_yAxis'].apply(lambda x: x.strip('[').strip(']').split(',')).values))))
            z = list(map(int, list(chain.from_iterable(df['accelerometer_milliG_zAxis'].apply(lambda x: x.strip('[').strip(']').split(',')).values))))
            temp = pd.DataFrame({'x' : x,'y' : y,'z' : z }).T
            acceleration.append(temp)
        else:
            continue 
    return np.array(acceleration)

def acceleration_data(train_dir = train_acceleration_data_path, deploy_dir = deploy_acceleration_data_path):
    return read_acceleration(train_dir), read_acceleration(deploy_dir)

def heart_data(train_dir = train_heart_data_path, deploy_dir = deploy_heart_data_path):
    return read_heart(train_dir), read_heart(deploy_dir)


def partition(x):
    if x == 'cycling':
        return 0
    elif x == 'lying':
        return 1
    elif x == 'running':
        return 2
    elif x == 'sitting':
        return 3
    elif x == 'stairsClimbing':
        return 4
    elif x == 'standing':
        return 5
    elif x == 'walking':
        return 6
    elif x == 'washingHands':
        return 7


def label_map():
    train_y = pd.read_csv(train_label_path, sep = ';', index_col=False).iloc[:,-1]
    deploy_y = pd.read_csv(deploy_label_path, sep = ';', index_col=False).iloc[:,-1]
    return np.array(train_y.map(partition)), np.array(deploy_y.map(partition))


def stft_generator(sig):
    main_temp = []
    for i in range(0,sig.shape[0],1):
        temp = []
        for j in range(0,sig.shape[1],1):
            f, t, Zxx  = stft(sig[i,j,:], 25, nperseg = 175)
            temp.append(np.abs(Zxx))
        main_temp.append(temp)
    return np.array(main_temp)


def contrastive_pair_generator(x, y, num_classes):
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]
    pairs = []
    labels = []

    for idx1 in range(x.shape[0]):
        # add a matching example
        x1 = x[idx1]
        label1 = y[idx1]
        idx2 = random.choice(digit_indices[label1])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [0]

        # add a non-matching example
        label2 = random.randint(0, num_classes - 1)
        while label2 == label1:
            label2 = random.randint(0, num_classes - 1)

        idx2 = random.choice(digit_indices[label2])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [1]

    return np.array(pairs), np.array(labels).astype("float32")

def euclidean_distance(vects):
    v1, v2 = vects
    sum_square = tf.math.reduce_sum(tf.math.square(v1 - v2), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))
