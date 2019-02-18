from RobustSTL import RobustSTL
from sample_generator import sample_generation

import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

def main():
    sample_list = sample_generation(total_len=750,
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
                        noise_std=0.316)
    
    result = RobustSTL(sample_list[0], 50, reg1=10.0, reg2= 0.5, K=2, H=5, dn1=1., dn2=1., ds1=50., ds2=1.)

    fig = plt.figure(figsize=(30,25))
    matplotlib.rcParams.update({'font.size': 22})
    samples = zip(result, ['sample', 'trend', 'seasonality', 'remainder'])

    for i, item in enumerate(samples):
        plt.subplot(4,1,(i+1))
        if i==0:
            plt.plot(item[0], color='blue')
            plt.title(item[1])
            plt.subplot(4,1,i+2)
            plt.plot(item[0], color='blue')
        else:
            plt.plot(item[0], color='red')
            plt.title(item[1])
    plt.show()
    
if __name__ == '__main__':
    main()