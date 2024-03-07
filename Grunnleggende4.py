# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 22:02:46 2023

@author: nilsk
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
import pandas as pd
import scipy.stats as stats
from scipy.stats import t
from scipy.stats import rankdata
from scipy.stats import wilcoxon
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import ttest_ind


df = 242
t_val = 1.74
p_val = np.round(1 - t.cdf(t_val, df), 3)

print(f"p-verdien er {p_val}")



x = np.arange(-3, 3, 0.1)
y = np.exp(-x**2 / 2) / np.sqrt(2*np.pi)

fig, ax = plt.subplots()
ax.plot(x, y, color='black')
ax.fill_between(x, y, where=x > 2.1, color='red')

ax.axvline(x=2.1, linestyle='dashed', color='blue')

ax.set(xlabel='Z', ylabel='',
       title='Standard normal distribution',
       ylim=(0, 0.45))
ax.grid(True)
plt.show()




z = 1.96
p_value = round(2 * (1 - norm.cdf(abs(z))), 2)
print("p-verdien er", p_value)





n1 = 126
n2 = 121
x1 = 57
x2 = 43
p = (x1 + x2) / (n1 + n2)
SE = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
z = (x1 / n1 - x2 / n2) / SE
print("z-verdien er", z)




df = pd.DataFrame({'x': np.arange(-3, 3, 0.001),
                   'y': np.nan})

df['y'] = df['x'].apply(lambda x: math.exp(-x**2 / 2) / math.sqrt(2*math.pi))

plt.plot(df['x'], df['y'], color='black')
plt.fill_between(df[df['x'] > z]['x'], df[df['x'] > z]['y'], color='red')
plt.fill_between(df[df['x'] < -z]['x'], df[df['x'] < -z]['y'], color='red')
plt.axvline(x=-z, linestyle='dashed', color='blue')
plt.axvline(x=z, linestyle='dashed', color='blue')
plt.xlabel('Z')
plt.ylabel('')
plt.title('Standard normalfordeling')
plt.ylim(0, 0.45)
plt.show()




z = z
p_value = 2 * (1 - stats.norm.cdf(abs(z)))
p_value



data = pd.DataFrame(data=[[57, 69],
                          [43, 78]],
                    index=["Molde", "Hamar"],
                    columns=["Svømmedyktig", "Ikke svømmedyktig"])
print(data)




data = np.array([[57, 126 - 57], [43, 121 - 43]])
statistic, p_value, _, _ = stats.chi2_contingency(data)
print("Kjikvadrattest :\n")
print(f"X^2: {statistic}")
print(f"Frihetsgrader: {data.shape[0] - 1}")
print(f"p-verdi: {p_value}")
print("Metode: Pearson's chi-squared test")




np.random.seed(321)

stille_lengde = pd.DataFrame({
    'id': range(1, 244),
    'lengde': np.random.normal(200, 35, 243)
})

summary = stille_lengde['lengde'].agg(['mean', 'std', 'min', 'max']).round(2).to_frame().T
summary.columns = ['snitt', 'sdavvik', 'korteste', 'lengste']

print(summary)






df = 242
t_val = 1.74
p_val = np.round(1 - t.cdf(t_val, df), 3)
print(f"p-verdien er {p_val}")





df = 242
t_val = 1.74
p_val = np.round(1 - t.cdf(t_val, df), 4)

x = np.linspace(-5, 5, 100)
y = t.pdf(x, df)

x_fill = np.linspace(t_val, 5, 100)
y_fill = t.pdf(x_fill, df)

plt.plot(x, y, color='black', linewidth=1.2)
plt.fill_between(x_fill, 0, y_fill, color='red', alpha=0.5)
plt.vlines(t_val, 0, np.max(y), linestyles='dashed', colors='blue')
plt.xticks(np.arange(-5, 6, 1))
plt.text(t_val + 1.3, t.pdf(t_val, df) + 0.02, f"p-verdi = {p_val}")
plt.xlabel('t-verdi')
plt.ylabel('')
plt.grid(True)
plt.show()





stille_lengde = np.random.normal(200, 35, 243)  # Replace this with your data
mu = 198
result2 = stats.wilcoxon(stille_lengde - mu)
print(f"WilcoxonResult statistic={result2.statistic}, pvalue={result2.pvalue}")




sinkdata = pd.read_csv("sinkdata.csv")
sinkdata["Differanse"] = sinkdata["Bunn"] - sinkdata["Overflate"]
print(sinkdata.head())




plt.figure(figsize=(10, 8))

cmap = plt.get_cmap('coolwarm')

sc = plt.scatter(sinkdata['Bunn'], sinkdata['Overflate'], c=sinkdata['Differanse'], cmap=cmap, s=60)

cbar = plt.colorbar(sc)
cbar.set_label('Differanse')

plt.title("Bunn vs Overflate\nDifferanse mellom bunn og overflate for samme punkt\nindikert med fargeskala")
plt.xlabel('Bunn')
plt.ylabel('Overflate')
plt.show()





sinkdata_summary = sinkdata["Differanse"].agg(
    snitt2 = lambda x: round(x.mean(), 2),
    sdavvik2 = lambda x: round(x.std(), 2),
    min2 = lambda x: round(x.min(), 2),
    max2 = lambda x: round(x.max(), 2)
)

print(sinkdata_summary)





result = stats.ttest_rel(sinkdata['Bunn'], sinkdata['Overflate'])
print(result)





x = np.linspace(-5, 5, 1000)
t_val = 4.8638
df = 9
p_val = 2 * (1 - t.cdf(abs(t_val), df))

plt.plot(x, t.pdf(x, df), linewidth=1)

plt.axvline(t_val, color='red', linestyle='dashed')
plt.axvline(-t_val, color='red', linestyle='dashed')

plt.text(t_val-1, 0.2, f"t = {t_val:.4f}")
plt.text(-t_val+1, 0.2, f"t = {-t_val:.4f}")
plt.text(t_val-1, 0.17, f"p = {p_val:.5f}")

plt.title('t-distribusjon')
plt.xlabel('t')
plt.ylabel('')
plt.grid(True)
plt.show()





differences = sinkdata['Bunn'] - sinkdata['Overflate']
ranked = rankdata(np.abs(differences))
statistic = np.sum(ranked[differences > 0])
print(f"Test statistic: {statistic}")
w, p = wilcoxon(differences)
print(f"Wilcoxon test result: W-statistic {w}, p-value {p}")





sys_bt2 = pd.read_csv('sys_bt2.csv')
print(sys_bt2.head(5))





summary = sys_bt2.groupby('RIAGENDR').agg(
    n=('RIAGENDR', 'count'),
    min=('BPXSY1', 'min'),
    median=('BPXSY1', 'median'),
    mean=('BPXSY1', 'mean'),
    max=('BPXSY1', 'max'),
    sd=('BPXSY1', 'std')
).reset_index()

print(summary)





palette1 = ["#0072B2", "#E69F00"]

plt.figure(figsize=(8, 6))
sns.boxplot(data=sys_bt2, x='RIAGENDR', y='BPXSY1', hue='RIAGENDR', palette=palette1)

plt.xticks(ticks=[0, 1], labels=["Kvinner", "Menn"])

plt.title("Boxplot for BPXSY1 for menn og kvinner")
plt.xlabel("Kjønn")
plt.ylabel("BPXSY1")
plt.legend(title="Gender")
sns.set_style("whitegrid")
plt.show()





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





grouped_data = sys_bt2.groupby('RIAGENDR')
test_statistics = []
p_values = []

for _, group in grouped_data:
    test_statistic, p_value = stats.shapiro(group['BPXSY1'])
    test_statistics.append(test_statistic)
    p_values.append(p_value)

results = pd.DataFrame({'Gender': grouped_data.groups.keys(), 'Test Statistic': test_statistics, 'p-value': p_values})

print(results)







male_data = sys_bt2[sys_bt2["RIAGENDR"] == "Male"]
female_data = sys_bt2[sys_bt2["RIAGENDR"] == "Female"]
ttest_likvar = ttest_ind(male_data["BPXSY1"], female_data["BPXSY1"], equal_var = True)
print(ttest_likvar)






df = len(sys_bt2) - 2
t_val = ttest_likvar.statistic

x = np.linspace(-9, 9, 1000)
y = t.pdf(x, df)

plt.plot(x, y)
plt.fill_between(x, y, where=np.logical_or(x >= t_val, x <= -t_val), color='red', alpha=0.3)

plt.title(f"t-distribution with degrees of freedom = {df}")
plt.xlabel("t-value")
plt.ylabel("Density")

plt.axvline(x=t_val, color='red', linestyle='dashed')
plt.axvline(x=-t_val, color='red', linestyle='dashed')

plt.text(5, 0.2, f"t = {t_val.round(3)}")
plt.text(-7, 0.2, f"t = -{t_val.round(3)}")
plt.text(4, 0.15, "p-value < 0.001")

plt.grid(True)
plt.show()





ttest_ulikvar = ttest_ind(male_data['BPXSY1'], female_data['BPXSY1'], equal_var=False)
print(ttest_ulikvar)





resultat5 = stats.mannwhitneyu(male_data['BPXSY1'], female_data['BPXSY1'])
resultat5


