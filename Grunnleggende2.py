# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 11:13:37 2023

@author: nilsk
"""

# Fordelingstyper - hvordan dataene “ser ut”

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import matplotlib.colors as mcolors
import statsmodels.api as sm
from scipy import stats
from scipy.stats import kstest
from scipy.stats import cramervonmises
from statsmodels.stats.stattools import jarque_bera
from scipy.stats import skew
from scipy.stats import binom
from collections import Counter
from scipy.stats import poisson
from scipy.stats import geom
from scipy.stats import expon


np.random.seed(30)
height = pd.DataFrame({
    'Gender': ['Man'] * 100,
    'Height': np.random.normal(179, 16, 100)
})


sns.set_style("whitegrid")
plt.figure(figsize=(8,6))
sns.histplot(height['Height'], bins=10, color="steelblue", edgecolor="black")
plt.xlabel('Height')
plt.ylabel('Count')
plt.show()



np.random.seed(30)
data = np.concatenate([
    np.random.choice(range(165, 176), 50, replace=True),
    np.random.choice(range(170, 181), 30, replace=True),
    np.random.choice(range(180, 186), 15, replace=True),
    np.random.choice(range(185, 191), 5, replace=True)
])
height2 = pd.DataFrame({
    'Gender': ['Man'] * len(data),
    'Height': data
})
sns.set_style("whitegrid")
plt.figure(figsize=(8,6))
sns.histplot(height2['Height'], bins=10, color="steelblue", edgecolor="black")
plt.xlabel('Height')
plt.ylabel('Count')
plt.show()



np.random.seed(321)
normal_distribution = pd.DataFrame({
    'f(x)': ['y'] * 10000,
    'x': np.random.normal(0, 1, 10000)
})
sns.set_style("whitegrid")
plt.figure(figsize=(8,6))
sns.histplot(normal_distribution['x'], bins=20, color="steelblue", edgecolor="black")
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()




np.random.seed(321)
normal_distribution = pd.DataFrame({
    'f(x)': ['y'] * 10000,
    'x': np.random.normal(0, 1, 10000)
})
sns.set_style("whitegrid")
plt.figure(figsize=(8,6))
sns.histplot(normal_distribution['x'], bins=30, color="steelblue", edgecolor="black", stat="density")
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, normal_distribution['x'].mean(), normal_distribution['x'].std())
plt.plot(x, p, 'k', linewidth=2)
plt.xlabel('x')
plt.ylabel('Density')
plt.show()



x = np.linspace(-4, 4, 1000)
plt.plot(x, norm.pdf(x), color='steelblue', linewidth=1.5)
plt.ylim(0, 0.5)
plt.yticks(np.arange(0, 0.6, 0.1))
plt.grid(True)
plt.show()


x = np.linspace(-4, 4, 200)
y = norm.pdf(x)
plt.plot(x, y, color='red', linewidth=1.5)
x_fill = np.linspace(-1, 1, 100)
y_fill = norm.pdf(x_fill)
plt.fill_between(x_fill, 0, y_fill, color='lightblue')
plt.axvline(-1, color='green', linewidth=1.5)
plt.axvline(1, color='green', linewidth=1.5)
plt.text(-1.35, 0.38, '-1 SD')
plt.text(1.3, 0.38, '1 SD')
plt.text(0, 0.2, '68%')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()




x = np.linspace(-4, 4, 200)
y = norm.pdf(x)
def plot_with_sd(low, high, color, text):
    plt.figure()
    plt.plot(x, y, color='red', linewidth=1.5)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    x_fill = np.linspace(low, high, 100)
    y_fill = norm.pdf(x_fill)
    plt.fill_between(x_fill, 0, y_fill, color=color)
    plt.axvline(low, color='green', linewidth=1.5)
    plt.axvline(high, color='green', linewidth=1.5)
    plt.text(low - 0.35, 0.38, f'{low} SD')
    plt.text(high + 0.1, 0.38, f'{high} SD')
    plt.text(0, 0.2, text)
    plt.show()
plot_with_sd(-2, 2, 'lightblue', '95%')
plot_with_sd(-3, 3, 'lightblue', '99.7%')





y_norm = np.random.normal(loc=0, scale=1, size=100000)
counts, bins = np.histogram(y_norm, bins=100)
fig, ax = plt.subplots()

for bin_start, bin_end, count in zip(bins[:-1], bins[1:], counts):
    # Choose color based on bin_start
    if bin_start < -3 or bin_start >= 3:
        color = 'white'
    elif bin_start < -2 or bin_start >= 2:
        color = 'green'
    elif bin_start < -1 or bin_start >= 1:
        color = 'red'
    else:
        color = 'blue'
    ax.bar(bin_start, count/(len(y_norm)*(bins[1]-bins[0])), width=bin_end-bin_start, color=color, align='edge')

ax.set_ylim([0, 0.6])
ax.set_title("Normalfordeling")
ax.set_xlabel("")

lwd = 2
lines = [
    ((2, -2), (0.48, 0.48), 'red'),
    ((3, -3), (0.55, 0.55), 'green'),
    ((1, -1), (0.41, 0.41), 'blue'),
    ((1, 1), (0, 0.41), 'blue'),
    ((-1, -1), (0, 0.41), 'blue'),
    ((2, 2), (0, 0.48), 'red'),
    ((-2, -2), (0, 0.48), 'red'),
    ((3, 3), (0, 0.55), 'green'),
    ((-3, -3), (0, 0.55), 'green')
]

for line in lines:
    ax.plot(line[0], line[1], color=line[2], linewidth=lwd)

ax.text(-0.2, 0.44, "68%", fontsize=12, color='blue')
ax.text(-0.2, 0.51, "95%", fontsize=12, color='red')
ax.text(-0.2, 0.58, "99.7%", fontsize=12, color='green')

plt.show()




qq_values = np.random.normal(size=10000)  # replace with actual data
quantiles = np.quantile(qq_values, np.linspace(0, 1, 100))
theoretical_quantiles = stats.norm.ppf(np.linspace(0, 1, 100))
plt.scatter(theoretical_quantiles, quantiles, s=6, color='blue')
plt.plot(theoretical_quantiles, theoretical_quantiles, color='red')
plt.title("Normal Q-Q plot")
plt.xlabel("Teoretisk forventning")
plt.ylabel("Data")
plt.show()



np.random.seed(90)
N = 5000
qqrightskew = pd.DataFrame({'value': np.random.negative_binomial(10, 0.1, N)})

plt.subplot(1, 2, 1)
plt.hist(qqrightskew['value'], color='steelblue', edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.subplot(1, 2, 2)
stats.probplot(qqrightskew['value'], dist='norm', plot=plt)
plt.title("Normal Q-Q plot")
plt.xlabel("Teoretisk forventning")
plt.ylabel("Data")
plt.tight_layout()
plt.show()



np.random.seed(91)
N = 5000
qqleftskew = np.random.beta(5, 1, size=N)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
axes[0].hist(qqleftskew, color='steelblue', edgecolor='black')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Histogram')
stats.probplot(qqleftskew, dist='norm', plot=axes[1])
plt.title("Normal Q-Q plot")
plt.xlabel("Teoretisk forventning")
plt.ylabel("Data")
plt.tight_layout()
plt.show()




np.random.seed(14)
N = 100
qqcauchy = np.random.standard_cauchy(size=N) * 5

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
axes[0].hist(qqcauchy, color='steelblue', edgecolor='black')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Histogram')
stats.probplot(qqcauchy, dist='norm', plot=axes[1])
plt.title("Normal Q-Q plot")
plt.xlabel("Teoretisk forventning")
plt.ylabel("Data")
plt.tight_layout()
plt.show()



np.random.seed(81)
qqlt = np.random.uniform(low=-1, high=1, size=1000)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
axes[0].hist(qqlt, color='steelblue', edgecolor='black')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Histogram')
stats.probplot(qqlt, dist='norm', plot=axes[1])
plt.title("Normal Q-Q plot")
plt.xlabel("Teoretisk forventning")
plt.ylabel("Data")
plt.tight_layout()
plt.show()




np.random.seed(10)
mode1 = np.random.normal(2, 1, size=50)
mode1 = mode1[mode1 > 0]
mode2 = np.random.normal(6, 1, size=50)
mode2 = mode2[mode2 > 0]
qqbimod = np.sort(np.concatenate((mode1, mode2)))

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
axes[0].hist(qqbimod, color='steelblue', edgecolor='black')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Histogram')
stats.probplot(qqbimod, dist='norm', plot=axes[1])
plt.title("Normal Q-Q plot")
plt.xlabel("Teoretisk forventning")
plt.ylabel("Data")
plt.tight_layout()
plt.show()






np.random.seed(10)

mode1 = np.random.normal(loc=2, scale=1, size=50)
mode1 = mode1[mode1 > 0]
mode2 = np.random.normal(loc=6, scale=1, size=50)
mode2 = mode2[mode2 > 0]
mode3 = np.random.normal(loc=10, scale=1, size=50)
mode3 = mode3[mode3 > 0]
qqmultimod = np.concatenate((mode1, mode2, mode3))

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

colors = ['steelblue', 'orange', 'green']
labels = ['Mode 1', 'Mode 2', 'Mode 3']
axes[0].hist([mode1, mode2, mode3], bins=10, color=colors, edgecolor='black', label=labels)
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Histogram')
axes[0].legend()

stats.probplot(qqmultimod, dist='norm', plot=axes[1])
axes[1].set_title("Normal Q-Q plot")
axes[1].set_xlabel("Theoretical Quantiles")
axes[1].set_ylabel("Sample Quantiles")

plt.tight_layout()
plt.show()




addata = pd.read_excel("Anderson-Darling_raw.xlsx")
result = stats.anderson(addata["Values"], dist="norm")
test_statistic = result.statistic
critical_values = result.critical_values
print(f"Test Statistic: {test_statistic}")
print("Critical Values:")
for level, crit_value in zip(result.significance_level, critical_values):
    print(f"    {level}%: {crit_value}")




addata5 = pd.read_csv("Anderson-Darling_raw.csv")
result = kstest(addata5["Values"], "norm")
print("KS Test Statistic:", result.statistic)
print("KS Test p-value:", result.pvalue)



W, p = stats.shapiro(addata5)
print("Shapiro-Wilk Test Statistic:", W)
print("Shapiro-Wilk Test p-value:", p)



mean = np.mean(addata["Values"])
std = np.std(addata["Values"])
addata_values_standardized = ((addata["Values"]) - mean) / std
result = cramervonmises(addata_values_standardized, 'norm')
print(f"CVM statistic: {result.statistic}")
print(f"p-value: {result.pvalue}")




result = jarque_bera(addata["Values"])
print(f"Jarque-Bera Test Statistic: {result[0]}")
print(f"p-value: {result[1]}")




Field_OLS_data = pd.read_excel("Field_datasett_OLS.xlsx")
df = pd.DataFrame({
    'x': ['Adverts'] * len(Field_OLS_data) + ['Sales'] * len(Field_OLS_data),
    'y': list(Field_OLS_data['Adverts']) + list(Field_OLS_data['Sales'])
})

plt.subplot(1, 2, 1)
plt.boxplot(df[df['x'] == 'Adverts']['y'], patch_artist=True, boxprops=dict(facecolor='#0072B2'))
plt.title('Adverts')
plt.ylabel('')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.boxplot(df[df['x'] == 'Sales']['y'], patch_artist=True, boxprops=dict(facecolor='#E69F00'))
plt.title('Sales')
plt.ylabel('')
plt.grid(True)

plt.tight_layout()
plt.show()





Karakterer = [1, 1, 1, 2, 2, 2, 5, 5, 5, 6, 6, 6]
plt.figure()
box_props = dict(facecolor='#006666', color='black')
plt.boxplot([Karakterer], patch_artist=True, boxprops=box_props)
plt.ylabel("Karakterer")
plt.xlabel("")
plt.title("Boksplott for karakterer")
plt.grid(True)
plt.show()




plt.figure()
plt.hist(Karakterer, bins=np.arange(min(Karakterer)-0.5, max(Karakterer)+1.5), edgecolor='black', color='#006666')
plt.xlabel("Karakterer")
plt.ylabel("Frekvens")
plt.title("Histogram for karakterer")
plt.grid(True)
plt.show()




plt.figure()
plt.scatter(Field_OLS_data["Adverts"], Field_OLS_data["Sales"], color='black')
plt.xlabel("Adverts")
plt.ylabel("Sales")
plt.title("Scatterplott for Adverts vs Sales")
plt.grid(True)
plt.show()





sns.scatterplot(x="Sales", y="Adverts", data=Field_OLS_data, color='orange')

# Smoothed regression line
lowess = sm.nonparametric.lowess(Field_OLS_data["Adverts"], Field_OLS_data["Sales"])
plt.plot(lowess[:, 0], lowess[:, 1], color='darkblue')

# Linear regression line
x = sm.add_constant(Field_OLS_data["Sales"])
model = sm.OLS(Field_OLS_data["Adverts"], x)
results = model.fit()
plt.plot(Field_OLS_data["Sales"], results.fittedvalues, color='red')

# Set labels and title
plt.xlabel("Sales")
plt.ylabel("Adverts")
plt.title("Scatter plot with Regression Lines")

# Show the plot
plt.show()




fig, ax = plt.subplots(2, 2, figsize=(8, 8))
sns.scatterplot(x="Sales", y="Adverts", data=Field_OLS_data, color='orange', ax=ax[0, 1])
ax[0, 1].set_xlabel("Sales")
ax[0, 1].set_ylabel("Adverts")

lowess = sm.nonparametric.lowess(Field_OLS_data["Adverts"], Field_OLS_data["Sales"])
ax[0, 1].plot(lowess[:, 0], lowess[:, 1], color='darkblue')

x = sm.add_constant(Field_OLS_data["Sales"])
model = sm.OLS(Field_OLS_data["Adverts"], x)
results = model.fit()
ax[0, 1].plot(Field_OLS_data["Sales"], results.fittedvalues, color='red')

sns.boxplot(x=Field_OLS_data["Adverts"], color='orange', ax=ax[0, 0])
ax[0, 0].set_xlabel("Adverts")
ax[0, 0].set_ylabel("")

sns.boxplot(x=Field_OLS_data["Sales"], color='orange', ax=ax[1, 1])
ax[1, 1].set_xlabel("Sales")
ax[1, 1].set_ylabel("")

ax[1, 0].spines['left'].set_visible(False)
ax[1, 0].spines['bottom'].set_visible(False)
ax[1, 0].spines['top'].set_visible(False)
ax[1, 0].spines['right'].set_visible(False)
ax[1, 0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

plt.tight_layout()
plt.show()




snitt_sim = 10
sd_sim = 5

sett1 = snitt_sim - (3 * sd_sim) - 5
sett2 = (2 * snitt_sim) + (3 * (2 * sd_sim)) + 5

m = np.linspace(sett1, sett2, 100000)

# plot
plt.plot(m, stats.norm.pdf(m, snitt_sim, sd_sim), color='#0072B2', label='Distribution 1')
plt.plot(m, stats.norm.pdf(m, snitt_sim, 2*sd_sim), color='#E69F00', label='Distribution 2')
plt.plot(m, stats.norm.pdf(m, 3*snitt_sim, 2*sd_sim), color='#000000', label='Distribution 3')
plt.legend(title='Color')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()




mammals = pd.read_csv('mammals.csv') 

# Create body weight histogram
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1) 
plt.hist(mammals['body'], color='steelblue', edgecolor='black')
plt.title('Body Weight')
plt.xlabel('Weight')
plt.ylabel('Frequency')

# Create brain weight histogram
plt.subplot(1, 2, 2) 
plt.hist(mammals['brain'], color='steelblue', edgecolor='black')
plt.title('Brain Weight')
plt.xlabel('Weight')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()




print(skew(mammals['body']))
print(skew(mammals['brain']))



plt.figure(figsize=(6, 6))
plt.scatter(mammals['body'], mammals['brain'])
plt.xlabel('Body')
plt.ylabel('Brain')
plt.title('Scatter plot of Body vs Brain')
plt.grid(True)
plt.show()




plt.figure(figsize=(6, 6))
plt.scatter(mammals['body'], mammals['brain'])
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Body Weight (log scale)')
plt.ylabel('Brain Weight (log scale)')
plt.title('Scatter plot of Body Weight vs Brain Weight')
plt.grid(True)
plt.show()




p = 0.3
sample = np.random.binomial(1, p, 100)
mean_value = np.mean(sample)
var_value = np.var(sample)
result = pd.DataFrame({"Sample Mean": [mean_value],
                       "Sample Variance": [var_value]})
print(result)




n = 20
p = 0.5
suksess = list(range(0, 21))
probability = binom.pmf(suksess, n, p)
df = pd.DataFrame({'Suksess': suksess, 'Sannsynlighet': probability})
plt.bar(df['Suksess'], df['Sannsynlighet'], color='steelblue', edgecolor='black', width=0.8)
plt.xlabel('Suksess')
plt.ylabel('Sannsynlighet')
plt.title('Binomial distribusjon (n = 20, p = 0.5)')
plt.grid(axis='y')
plt.show()





n = 20
p = 0.2
suksess = list(range(0, 21))
sannsynlighet = binom.pmf(suksess, n, p)
df = pd.DataFrame({'Suksess': suksess, 'Sannsynlighet': sannsynlighet})
plt.bar(df['Suksess'], df['Sannsynlighet'], color='steelblue', width=0.5)
plt.xlabel('Suksess')
plt.ylabel('Sannsynlighet')
plt.title('Binomial distribusjon (n = 20, p = 0.2)')
plt.grid(axis='y')
plt.show()




np.random.seed(32)
terning10 = np.random.choice(np.arange(1, 7), size=10)
x = []
y = []
for i in range(1, 7):
    occur = np.sum(terning10 == i)
    for j in range(occur):
        x.append(i)
        y.append(j + 1)
plt.scatter(x, y, color='steelblue')
plt.xlabel('Verdi på terning')
plt.yticks(range(1, max(y)+1))
plt.title('10 terningkast')
plt.show()




np.random.seed(32)
terning10 = np.random.choice(np.arange(1, 7), size=100)
x = []
y = []
for i in range(1, 7):
    occur = np.sum(terning10 == i)
    for j in range(occur):
        x.append(i)
        y.append(j + 1)
plt.scatter(x, y, color='steelblue')
plt.xlabel('Verdi på terning')
plt.yticks(range(1, max(y)+1))
plt.title('100 terningkast')
plt.show()




np.random.seed(43)
terning_runde1 = np.random.choice(np.arange(1, 7), size=600)
counts = Counter(terning_runde1)
for k in sorted(counts.keys()):
    print(f"{k}: {counts[k]}")
    
    

np.random.seed(44)
terning_runde2 = np.random.choice(np.arange(1, 7), size=600)
counts = Counter(terning_runde1)
for k in sorted(counts.keys()):
    print(f"{k}: {counts[k]}")
    
    
    
np.random.seed(45)
terning_runde3 = np.random.choice(np.arange(1, 7), size=600)
counts = Counter(terning_runde1)
for k in sorted(counts.keys()):
    print(f"{k}: {counts[k]}")
    
    
    

np.random.seed(43)
terning_runde4 = np.random.choice(np.arange(1, 7), size=6000000)
counts = Counter(terning_runde1)
for k in sorted(counts.keys()):
    print(f"{k}: {counts[k]}")
    
    
    
    
plt.hist(terning_runde4, bins=6, range=(0.5, 6.5), color='steelblue', edgecolor='black')
plt.title('Søylediagram for 6 000 000 terningkast')
plt.xlabel('Verdi på terning')
plt.ylabel('Antall')
plt.show()




x = np.arange(0, 51)

lambdas = [5, 10, 20]
colors = ["blue", "red", "darkgreen"]
labels = ["λ = 5", "λ = 10", "λ = 20"]

data = pd.DataFrame()

for i, lamb in enumerate(lambdas):
    temp_df = pd.DataFrame({
        'x': x,
        'y': poisson.pmf(x, lamb),
        'lambda': lamb
    })
    data = pd.concat([data, temp_df])

plt.figure(figsize=(10, 6))
sns.barplot(x='x', y='y', hue='lambda', data=data, palette=colors, dodge=True)
plt.xlabel('Antall hendelser')
plt.ylabel('Sannsynlighet')
plt.title('Poisson sannsynlighetsfordeling')
plt.legend(title="λ", title_fontsize = '13', labels=labels)
plt.show()



maal = np.arange(0, 11)
df = pd.DataFrame({'maal': maal, 'poisson': poisson.pmf(maal, mu=2.5)})

plt.figure(figsize=(10, 6))
plt.bar(df['maal'], df['poisson'], color='steelblue', edgecolor='black')
plt.xlabel('# Mål', fontsize=13, fontweight='bold')
plt.ylabel('Sannsynlighet', fontsize=13, fontweight='bold')
plt.title('Poissonfordeling (lambda = 2.5)', fontsize=15)
plt.grid(axis='y')
plt.show()




x_dgeom = np.arange(1, 21)
y_dgeom = geom.pmf(x_dgeom, p=0.4)

df = pd.DataFrame({'x': x_dgeom, 'y': y_dgeom})

plt.figure(figsize=(10, 6))
plt.plot(df['x'], df['y'], color='steelblue', linewidth=1.5)
plt.title('Geometrisk fordeling for p = 0.4')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.show()



eksford = np.linspace(0, 20, 1000)
rates = [0.2, 1.0, 1.5, 2.0]

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

for ax, rate in zip(axs.flatten(), rates):
    df = pd.DataFrame({'x': eksford, 'fx': expon.pdf(eksford, scale=1/rate)})
    ax.plot(df['x'], df['fx'])
    ax.set_title(f'lambda = {rate}')
    ax.grid(True)

plt.tight_layout()
plt.show()






