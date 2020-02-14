import pandas as pd
import numpy as np
from MetaFeatures.distance_mf import get_distance_metafeatures
from MetaFeatures.general_mf import  get_statlog_metafeatures
from MetaFeatures.landmark_mf import get_kmeans_meta_features, get_landmarking
from Utils.preprocessing import preprocess
from Utils.file_loader import get_dataframes, get_generated_dataframes
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

def create_mf():
    dataframes = get_dataframes()
    generated_dataframes = get_generated_dataframes()
    # Not using all dataframes
    total_dfs = dataframes
    dataframes = [(x, pd.read_csv(x, header=None)) for x in total_dfs]
    processed_df = [(x[0], preprocess(x[1])) for x in dataframes]

    # start = time.time()
    # statlog = pd.DataFrame.from_records([get_statlog_metafeatures(x[0], x[1]) for x in dataframes])
    # end = time.time()
    # print("Time to generate statlog features:", end-start)

    start = time.time()
    distance, d_time_stats = [],[]
    for x in processed_df:
        r, t = get_distance_metafeatures(x[0], x[1])
        distance.append(r)
        d_time_stats.append(t)
    distance = pd.DataFrame.from_records(distance)
    end = time.time()
    print("Time to generate distance features:", end-start)

    # start = time.time()
    # kmeans = pd.DataFrame.from_records([get_kmeans_meta_features(x[0], x[1]) for x in processed_df])
    # end = time.time()
    # print ("time to generate KMeans features:", end-start)

    start = time.time()
    landmarks, l_time_stats = [],[]
    for x in processed_df:
        r, t = get_landmarking(x[0], x[1])
        landmarks.append(r)
        l_time_stats.append(t)
    landmarks = pd.DataFrame.from_records(landmarks)
    end = time.time()
    print("Time to generate landmark features:", end-start)
  
    # statlog.to_csv('./statlogMF.csv', index=False)
    # distance.to_csv('./distanceMF.csv', index=False)
    # kmeans.to_csv('./kmeansMF.csv', index=False)
    # landmarks.to_csv('./landmarkMF.csv', index=False)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter([x[0] for x in d_time_stats], [x[1] for x in d_time_stats])
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Number of attributes")
    # X = []
    # Y = []
    # Z =[]
    # for index in range(len(d_time_stats)):
    #     X.append(d_time_stats[index][0])
    #     Y.append(d_time_stats[index][1])
    #     Z.append(d_time_stats[index][2])
    # d = ax.bar3d(X, Y, [0]*len(X), [7] * len(X), [7] * len(Y), Z, label='Distance based')
    # d._facecolors2d=d._facecolors3d
    # d._edgecolors2d=d._edgecolors3d
    # # ax.scatter(X, Y, Z, label='Distance based', marker='o')

    # X = []
    # Y = []
    # Z =[]
    # for index in range(len(l_time_stats)):
    #     X.append(l_time_stats[index][0])
    #     Y.append(l_time_stats[index][1])
    #     Z.append(l_time_stats[index][2])

    # l = ax.bar3d(X, Y, [0]*len(X), [7] * len(X), [7] * len(Y), Z, label='Landmark based')
    # l._facecolors2d=l._facecolors3d
    # l._edgecolors2d=l._edgecolors3d
    # # ax.scatter(X, Y, Z, label='Landmark based', marker='^')


    # ax.set_xlabel("No. samples")
    # ax.set_ylabel("No. attributes")
    # ax.set_zlabel("Time (s)")

    # plt.legend()
    plt.savefig('./datasets.png', format='png')

if __name__ == "__main__":
    create_mf()
