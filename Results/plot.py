import pandas as pd
import matplotlib.pyplot as plt
import os
import sys


def plot_learning_curve(filename):
    df = pd.read_csv(filename)
    plt.plot(df['episode'],df['score'],label='score',color='blue',linewidth=0.5)
    plt.plot(df['episode'],df['score_avg'],label='score_avg',color='red',linewidth=1)
    plt.title('Learning Curve')
    plt.legend(labels=['Score','Averages Score over 100 episodes'])
    plt.xlabel('Episode')
    plt.ylabel('Score')
    # plt.savefig(filename)
    plt.show()
    
def plot_learning_curve_abs(filename):
    df = pd.read_csv(filename)
    plt.plot(df['episode'],df['score'])
    plt.title('Running average of previous 100 scores')
    # plt.savefig(filename)
    plt.show()





def main(argv):
    path = argv[1]
    filename = os.path.join(os.getcwd(),path)
    plot_learning_curve(filename)
    
if __name__ == "__main__":
    main(sys.argv)
    