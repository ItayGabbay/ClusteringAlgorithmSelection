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

    X = list(range(1,11))
    ydist = []
    ylmrk = []
    for k in range(1,11):
        ydist.append(evaluate(distance, rankings, k)[0][0])
        ylmrk.append(evaluate(landmark, rankings, k)[0][0])

    plt.plot(X, ydist)
    plt.plot(X, ylmrk)
    plt.show()
    metrics = [("BP", pd.read_csv('./result_BP.csv')),
                ("CH", pd.read_csv('./result_CH.csv')),
                ("DB", pd.read_csv('./result_DB.csv')),
                ("DU", pd.read_csv('./result_DU.csv')),
                ("HKK", pd.read_csv('./result_HKK.csv')),
                ("HL", pd.read_csv('./result_HL.csv')),
                ("MC", pd.read_csv('./result_MC.csv')),
                ("SIL", pd.read_csv('./result_SIL.csv'))]
    
    # for name, rk in metrics:
    #     print("Distance", name, ":", evaluate(distance, rk))
    #     print("Landmark", name, ":", evaluate(landmark, rk))

if __name__ == "__main__":
    main()