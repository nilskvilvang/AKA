# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 20:34:05 2023

@author: nilsk
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.stats import probplot
from scipy.stats import t
from scipy.stats import shapiro
from scipy.stats import levene



def alphabeta(mean_sick=100, sd_sick=10, mean_healthy=130, sd_healthy=10, cutoff=120, n=10000, side="below", do_plot=True):
    popsick = np.random.normal(mean_sick, sd_sick, n)
    pophealthy = np.random.normal(mean_healthy, sd_healthy, n)
    
    if side == "below":
        truepos = np.sum(popsick <= cutoff)
        falsepos = np.sum(pophealthy <= cutoff)
        trueneg = np.sum(pophealthy > cutoff)
        falseneg = np.sum(popsick > cutoff)
    elif side == "above":
        truepos = np.sum(popsick >= cutoff)
        falsepos = np.sum(pophealthy >= cutoff)
        trueneg = np.sum(pophealthy < cutoff)
        falseneg = np.sum(popsick < cutoff)
    
    spec = trueneg / (trueneg + falsepos)
    alpha = 1 - spec
    sens = pow = truepos / (truepos + falseneg)
    beta = 1 - sens
    
    pos_pred = truepos / (truepos + falsepos)
    neg_pred = trueneg / (trueneg + falseneg)
    
    if do_plot:
        plt.figure(figsize=(10,6))
        plt.hist(popsick, bins=30, alpha=0.5, label="sick", color="#E69F00")
        plt.hist(pophealthy, bins=30, alpha=0.5, label="healthy", color="#0072B2")
        plt.axvline(cutoff, color='r', linestyle='dashed', linewidth=2)
        plt.legend(loc='upper right')
        plt.show()

    return alpha, beta

alpha, beta = alphabeta(mean_sick=100, sd_sick=10, mean_healthy=250, sd_healthy=15, cutoff=160, n=100000, do_plot=True, side="below")
print(alpha, beta)



alpha_beta_2 = alphabeta(mean_sick=100, sd_sick=10, mean_healthy=140, sd_healthy=15, cutoff=120, n=100000, do_plot=True, side="below")
print(alpha_beta_2)



alpha_beta_3 = alphabeta(mean_sick=100, sd_sick=10, mean_healthy=140, sd_healthy=15, cutoff=105, n=100000, do_plot=True, side="below")
print(alpha_beta_3)



cutoffs = np.arange(0, 200.1, 0.1)
plot_frame = [alphabeta(mean_sick=100, sd_sick=10, mean_healthy=140, sd_healthy=15, cutoff=cutoff, n=50000, do_plot=False, side="below") for cutoff in cutoffs]
alpha_values, beta_values = zip(*plot_frame)
plt.figure(figsize=(10, 6))
plt.plot(cutoffs, alpha_values, label='alpha', linewidth=2)
plt.plot(cutoffs, beta_values, label='beta', linestyle='--', color='steelblue', linewidth=2)
plt.xlabel('Cut-off value')
plt.ylabel('Alpha/Beta')
plt.legend(loc='upper left')
plt.show()




np.random.seed(123)

mean_set = np.random.normal(50000, 1000, 50)
sample = np.arange(1, 51, 1)
conf_level = np.repeat([90, 95, 99], 50)
mean = np.repeat(mean_set, 3)
upper = mean + np.repeat([1677, 2010, 2680], 50)
lower = mean - np.repeat([1677, 2010, 2680], 50)
capture = np.where((lower < 50000) & (upper > 50000), 1, 0)

ci_data = pd.DataFrame({'Sample': np.tile(sample, 3),
                        'Mean': mean,
                        'Conf_level': conf_level,
                        'upper': upper,
                        'lower': lower,
                        'Capture': capture})

ci_data['Capture'] = ci_data['Capture'].astype('category')

# Create a color mapping for the capture column
colors = {0: 'red', 1: 'black'}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Iterate over each confidence level
for i, conf in enumerate([90, 95, 99]):
    data = ci_data[ci_data['Conf_level'] == conf]
    axes[i].scatter(data['Mean'], data['Sample'], c=data['Capture'].map(colors))
    axes[i].hlines(data['Sample'], data['lower'], data['upper'], colors=data['Capture'].map(colors))
    axes[i].axvline(x=50000, linestyle='--', color='blue')
    axes[i].set_xlim([40000, 60000])
    axes[i].set_ylim([0, 51])
    axes[i].set_title(f'Conf_level: {conf}')

plt.tight_layout()
plt.show()







np.random.seed(42)
n_simulations = 1000
min_sample_size = 10
max_sample_size = 100
step = 10

def confidence_interval(sample):
    mean_sample = np.mean(sample)
    ci_width = stats.t.ppf(0.975, len(sample) - 1) * np.std(sample, ddof=1) / np.sqrt(len(sample))
    return np.array([mean_sample - ci_width, mean_sample + ci_width])

results = []
for sample_size in range(min_sample_size, max_sample_size + 1, step):
    ci_width_sum = 0
    mean_sum = 0
    
    for i in range(n_simulations):
        sample = np.random.normal(size=sample_size)
        ci = confidence_interval(sample)
        ci_width = ci[1] - ci[0]
        mean_sum += np.mean(sample)
        ci_width_sum += ci_width
        
    results.append([sample_size, mean_sum / n_simulations, ci_width_sum / n_simulations])

df = pd.DataFrame(results, columns=['sample_size', 'mean', 'ci_width'])

fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(df['mean'], df['sample_size'], xerr=df['ci_width']/2, fmt='o', color='black', capsize=5)
ax.set_xlim([-1, 1])
ax.set_ylim([0, max_sample_size])
ax.set_xticks(np.arange(-1, 1.1, 0.1))
ax.set_yticks(np.arange(0, max_sample_size + 1, step))
ax.set_xlabel('Estimated Mean')
ax.set_ylabel('Sample Size')
ax.set_title('Confidence Limits vs Sample Size')
ax.text(-0.9, 90, 'Confidence level: 95%', fontsize=12)
plt.show()




import numpy as np
from statsmodels.stats.proportion import proportions_ztest

count = np.array([57, 43])
nobs = np.array([126, 121])

# Perform the test
stat, pval = proportions_ztest(count, nobs, alternative='two-sided', prop_var=False)

# Calculate the confidence interval for the difference in proportions
prop1 = count[0] / nobs[0]
prop2 = count[1] / nobs[1]
se_diff = np.sqrt(prop1 * (1 - prop1) / nobs[0] + prop2 * (1 - prop2) / nobs[1])
diff = prop1 - prop2
margin_of_error = 1.96 * se_diff
conf_int = diff - margin_of_error, diff + margin_of_error

print("Confidence Interval:", conf_int)





np.random.seed(321)

stille_lengde = pd.DataFrame({'id': np.arange(1, 244),
                              'lengde': np.random.normal(200, 35, 243)})

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Histogram
axes[0].hist(stille_lengde['lengde'], color='steelblue', edgecolor='black')
axes[0].set_xlabel('Length')
axes[0].set_ylabel('Count')

# Q-Q Plot
probplot(stille_lengde['lengde'], plot=axes[1])
axes[1].set_title('Q-Q Plot')

plt.tight_layout()
plt.show()




round(stille_lengde['lengde'].mean(), 2)




qt_verdi = stats.t.ppf(1 - 0.05/2, df=242)
round(qt_verdi, 2)




sd_value = np.std(stille_lengde['lengde'], ddof=1)
round(sd_value, 2)



sinkdata = pd.read_excel("zinc_conc.xlsx")
sinkdata['Differanse'] = sinkdata['Bunn'] - sinkdata['Overflate']
print(sinkdata)





gjennomsnitt = sinkdata['Differanse'].mean()
sd = round(sinkdata['Differanse'].std(), 4)
t_verdi = round(t.ppf(0.05 / 2, df=9), 4)
print(f"Gjennomsnittsdifferansen er {gjennomsnitt}")




print(f"Standardavviket for gjennomsnittsdifferansen er {sd}")




print(f"T-verdien er {t_verdi}")




margin_error = t_verdi * (sd / np.sqrt(sinkdata.shape[0]))
grenser = np.round([gjennomsnitt - margin_error, gjennomsnitt + margin_error], 4)
print("Konfidensintervallet er (", grenser[1], ",", grenser[0], ")")




sys_bt2 = pd.read_csv("sys_bt2.csv")
sys_bt2 = sys_bt2[['RIAGENDR', 'BPXSY1']].dropna()




summary_stats = sys_bt2.groupby('RIAGENDR').agg(
    n=('BPXSY1', 'size'),
    min=('BPXSY1', 'min'),
    q1=('BPXSY1', lambda x: np.percentile(x, 25)),
    median=('BPXSY1', 'median'),
    mean=('BPXSY1', 'mean'),
    q3=('BPXSY1', lambda x: np.percentile(x, 75)),
    max=('BPXSY1', 'max'),
    IQR=('BPXSY1', lambda x: np.percentile(x, 75) - np.percentile(x, 25)),
    sd=('BPXSY1', 'std')
).reset_index()
summary_stats




shapiro_results = sys_bt2.groupby('RIAGENDR')['BPXSY1'].apply(lambda x: shapiro(x)[1]).reset_index(name='p_value')
shapiro_results




import statsmodels.api as sm
import matplotlib.pyplot as plt

# Create separate datasets for Male and Female
male_data = sys_bt2[sys_bt2["RIAGENDR"] == "Male"]["BPXSY1"]
female_data = sys_bt2[sys_bt2["RIAGENDR"] == "Female"]["BPXSY1"]

# Create QQ plots
qq_male = sm.ProbPlot(male_data)
qq_female = sm.ProbPlot(female_data)

# Plot the QQ plots side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Plot QQ plot for Male data
sm.qqplot(male_data, fit=True, line="45", ax=axes[0])
axes[0].set_title("QQ Plot - Male")

# Plot QQ plot for Female data
sm.qqplot(female_data, fit=True, line="45", ax=axes[1])
axes[1].set_title("QQ Plot - Female")

plt.tight_layout()
plt.show()






levene_test = levene(sys_bt2['BPXSY1'][sys_bt2['RIAGENDR'] == 'Male'],
                     sys_bt2['BPXSY1'][sys_bt2['RIAGENDR'] == 'Female'])

print("Levene's test statistic:", levene_test.statistic)
print("Levene's test p-value:", levene_test.pvalue)




t_quantile = abs(t.ppf(0.05 / 2, df=66))
t_quantile



t_quantile2 = abs(t.ppf(0.05 / 2, df=6666))
t_quantile2
