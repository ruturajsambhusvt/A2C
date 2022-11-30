import matplotlib.pyplot as plt 
import numpy as np
import csv
import os
import time
import datetime

def plotLearning(scores, title, filename, x=None, window=5):   
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])

    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Rewards')       
    plt.xlabel('Episodes')     
    plt.plot(x, scores,label='Score',linewidth=0.5)              
    plt.plot(x, running_avg,label='Running Average',linewidth=2)
    plt.legend()
    plt.title(title)
    plt.savefig(filename)
    plt.show()

class write_data():

    def __init__(self,name,fieldnames=None) -> None:
    #    self.episode = episode
    #    self.score = score
    #    self.score_avg = score_avg
       self.name = name+str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))+".csv"
       if fieldnames is not None:
           self.fieldnames = fieldnames
       else:
           self.fieldnames = ["episode","score","score_avg"]
    
       with open(os.path.join(os.getcwd()+str(self.name)), 'w') as file:
           writer =csv.DictWriter(file,fieldnames=self.fieldnames)
           writer.writeheader()
           
       
    def write(self,episode,score,score_avg):
        with open(os.path.join(os.getcwd()+str(self.name)), 'a') as file:
            writer = csv.DictWriter(file,fieldnames=self.fieldnames)
            info = {"episode": episode,"score":score,"score_avg":score_avg}
            writer.writerow(info)
