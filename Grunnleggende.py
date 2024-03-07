# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 08:29:05 2023

@author: nilsk
"""

# Standardisering - transformasjon av data (z-skåre)
import scipy.stats as stats

popgjsnitt = 167
sd = 6
probability = 1 - stats.norm.cdf(175, loc=popgjsnitt, scale=sd)
print(probability)

result = stats.norm.cdf(175, loc=popgjsnitt, scale=sd) - stats.norm.cdf(165, loc=popgjsnitt, scale=sd)
print(result)

# Modeller og modellering

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

mpg = pd.read_csv("mpg.csv")
sns.set(style="darkgrid")
sns.scatterplot(data=mpg, x="displ", y="hwy")
lowess_results = lowess(mpg['hwy'], mpg['displ'], frac=0.2)
plt.plot(lowess_results[:, 0], lowess_results[:, 1], color='red')
plt.xlabel("displ")
plt.ylabel("hwy")
plt.title("Scatter Plot with Smooth Curve")
plt.show()



