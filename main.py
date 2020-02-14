from Evaluator.evaluation import evaluate, get_baseline
import pandas as pd
import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt

def main():

    distance = pd.read_csv('./distanceMF.csv')
    landmark = pd.read_csv('./landmarkMF.csv')

    rankings = pd.read_csv('./rankings.csv')
    rankings[rankings.drop('dataset', axis=1).columns] = rankings.drop('dataset', axis=1).rank(axis=1, method='average')

    landmark_distance = pd.merge(distance, landmark)

    rk_baseline = rankings.loc[rankings['dataset'].isin(landmark['dataset'])]

    print("Standard Rankings:", get_baseline([6,7,1,2,4,5,3], rk_baseline))

    print("Distance Features:", evaluate(distance, rankings))

    print("Landmark features:", evaluate(landmark, rankings, 9))
    print("Landmark + Distance features:", evaluate(landmark_distance, rankings))

if __name__ == "__main__":
    main()