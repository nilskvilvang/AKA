# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 14:11:37 2023

@author: nilsk
"""

import pygraphviz as pgv
from IPython.display import Image
from tabulate import tabulate
import numpy as np


G = pgv.AGraph(directed=True)
G.add_node('A')
G.add_node('Y')
G.add_edge('A', 'Y')
G.layout(prog='dot')
G.draw('graph.png')
Image(filename='graph.png')




G = pgv.AGraph(directed=True)
G.add_node('A', shape='plaintext', label='B')
G.add_node('Y', shape='plaintext', label='Y')
G.add_node('C', shape='plaintext', label='A')
G.add_edge('A', 'Y')
G.add_edge('C', 'A')
G.add_edge('C', 'Y')
G.layout(prog='dot')
G.draw('graph.png')
Image(filename='graph.png')




G = pgv.AGraph(directed=True)
G.add_node('A', shape='plaintext', label='A')
G.add_node('B', shape='plaintext', label='B')
G.add_node('C', shape='plaintext', label='C')
G.add_node('D', shape='plaintext', label='D')
G.add_node('Y', shape='plaintext', label='Y')
G.add_edge('A', 'B')
G.add_edge('B', 'Y')
G.add_edge('C', 'D')
G.add_edge('D', 'B')
G.add_subgraph(['A', 'B', 'Y'], rank='same')
G.add_subgraph(['C', 'D'], rank='same')
G.layout(prog='dot')
G.draw('graph.png')
Image(filename='graph.png')





G = pgv.AGraph(directed=True)
G.add_node('A', shape='plaintext', label='Having cats')
G.add_node('Y', shape='plaintext', label='Happiness')
G.add_node('Lik', shape='plaintext', label='Liking cats')
G.add_node('Pers', shape='plaintext', label='Personality')
G.add_node('Lon', shape='plaintext', label='Loneliness')
G.add_node('SEP', shape='box', label='Socioeconomic position')
G.add_edge('A', 'Y')
G.add_edge('Lik', 'A')
G.add_edge('Pers', 'Lik')
G.add_edge('Pers', 'Y')
G.add_edge('A', 'Lon')
G.add_edge('Lon', 'Y')
G.add_edge('SEP', 'A')
G.add_edge('SEP', 'Y')
G.add_subgraph(['A', 'Y'], rank='same')
G.layout(prog='dot')
G.draw('graph.png')
Image(filename='graph.png')





G = pgv.AGraph(directed=True)
G.add_node('A', shape='plaintext', label='A')
G.add_node('B', shape='plaintext', label='B')
G.add_node('C', shape='plaintext', label='C')
G.add_edge('A', 'B')
G.add_edge('B', 'C')
G.add_subgraph(['A', 'B', 'C'], rank='same')
G.layout(prog='dot')
G.draw('graph.png')
Image(filename='graph.png')





G = pgv.AGraph(directed=True)
G.add_node('A', shape='plaintext', label='A')
G.add_node('B', shape='plaintext', label='B')
G.add_node('X', shape='plaintext', label='X')
G.add_edge('A', 'B')
G.add_edge('X', 'A')
G.add_edge('X', 'B')
G.add_subgraph(['A', 'B'], rank='same')
G.layout(prog='dot')
G.draw('graph.png')
Image(filename='graph.png')





G = pgv.AGraph(directed=True)
G.add_node('A', shape='plaintext', label='A')
G.add_node('B', shape='plaintext', label='B')
G.add_node('X', shape='plaintext', label='X')
G.add_edge('A', 'B')
G.add_edge('X', 'B')
G.add_subgraph(['A', 'B'], rank='same')
G.layout(prog='dot')
G.draw('graph.png')
Image(filename='graph.png')




import numpy as np
from tabulate import tabulate

# Set seed
np.random.seed(1234)

# Generate data for treatment group
age_treatment = np.random.normal(loc=60, scale=10, size=243)
sex_treatment = np.random.choice(["M", "F"], size=243, replace=True, p=[0.4, 0.6])
sbp_treatment = np.random.normal(loc=140, scale=15, size=243)
smoking_treatment = np.random.choice(["Y", "N"], size=243, replace=True, p=[0.25, 0.75])
exercise_treatment = np.random.choice(["Y", "N"], size=243, replace=True, p=[0.6, 0.4])

# Generate data for control group
age_control = np.random.normal(loc=50, scale=10, size=1432)
sex_control = np.random.choice(["M", "F"], size=1432, replace=True, p=[0.5, 0.5])
sbp_control = np.random.normal(loc=130, scale=10, size=1432)
smoking_control = np.random.choice(["Y", "N"], size=1432, replace=True, p=[0.4, 0.6])
exercise_control = np.random.choice(["Y", "N"], size=1432, replace=True, p=[0.7, 0.3])

# Calculate standardized mean differences (SMD)
smd_age = (np.mean(age_treatment) - np.mean(age_control)) / np.sqrt((np.var(age_treatment) + np.var(age_control)) / 2)
smd_sex = (np.mean(sex_treatment == "M") - np.mean(sex_control == "M")) / np.sqrt((np.var(sex_treatment == "M") + np.var(sex_control == "M")) / 2)
smd_sbp = (np.mean(sbp_treatment) - np.mean(sbp_control)) / np.sqrt((np.var(sbp_treatment) + np.var(sbp_control)) / 2)
smd_smoking = (np.mean(smoking_treatment == "Y") - np.mean(smoking_control == "Y")) / np.sqrt((np.var(smoking_treatment == "Y") + np.var(smoking_control == "Y")) / 2)
smd_exercise = (np.mean(exercise_treatment == "Y") - np.mean(exercise_control == "Y")) / np.sqrt((np.var(exercise_treatment == "Y") + np.var(exercise_control == "Y")) / 2)

# Create the table
data = [
    ["n", "Alder", "Kjønn", "SBP (mmHg)", "Røyking", "Trening"],
    [len(age_treatment), round(np.mean(age_treatment), 1), "/".join(np.unique(sex_treatment, return_counts=True)[1].astype(str)), round(np.mean(sbp_treatment), 2), "/".join(np.unique(smoking_treatment, return_counts=True)[1].astype(str)), "/".join(np.unique(exercise_treatment, return_counts=True)[1].astype(str))],
    [len(age_control), round(np.mean(age_control), 1), "/".join(np.unique(sex_control, return_counts=True)[1].astype(str)), round(np.mean(sbp_control), 2), "/".join(np.unique(smoking_control, return_counts=True)[1].astype(str)), "/".join(np.unique(exercise_control, return_counts=True)[1].astype(str))]
]

# Print the table
table = tabulate(data, headers="firstrow", tablefmt="pipe")
print(table)




np.random.seed(1234)
age_treatment = np.random.normal(loc=60, scale=10, size=243)
sex_treatment = np.random.choice(["M", "F"], size=243, replace=True, p=[0.4, 0.6])
sbp_treatment = np.random.normal(loc=140, scale=15, size=243)
smoking_treatment = np.random.choice(["Y", "N"], size=243, replace=True, p=[0.25, 0.75])
exercise_treatment = np.random.choice(["Y", "N"], size=243, replace=True, p=[0.6, 0.4])
age_control = np.random.normal(loc=50, scale=10, size=1432)
sex_control = np.random.choice(["M", "F"], size=1432, replace=True, p=[0.5, 0.5])
sbp_control = np.random.normal(loc=130, scale=10, size=1432)
smoking_control = np.random.choice(["Y", "N"], size=1432, replace=True, p=[0.4, 0.6])
exercise_control = np.random.choice(["Y", "N"], size=1432, replace=True, p=[0.7, 0.3])
smd_age = (np.mean(age_treatment) - np.mean(age_control)) / np.sqrt((np.var(age_treatment) + np.var(age_control)) / 2)
smd_sex = (np.mean(sex_treatment == "M") - np.mean(sex_control == "M")) / np.sqrt((np.var(sex_treatment == "M") + np.var(sex_control == "M")) / 2)
smd_sbp = (np.mean(sbp_treatment) - np.mean(sbp_control)) / np.sqrt((np.var(sbp_treatment) + np.var(sbp_control)) / 2)
smd_smoking = (np.mean(smoking_treatment == "Y") - np.mean(smoking_control == "Y")) / np.sqrt((np.var(smoking_treatment == "Y") + np.var(smoking_control == "Y")) / 2)
smd_exercise = (np.mean(exercise_treatment == "Y") - np.mean(exercise_control == "Y")) / np.sqrt((np.var(exercise_treatment == "Y") + np.var(exercise_control == "Y")) / 2)
data = [
    ["Variabel", "Behandling", "Kontroll", "SMD"],
    ["n", len(age_treatment), len(age_control), ""],
    ["Alder", round(np.mean(age_treatment), 1), round(np.mean(age_control), 1), round(smd_age, 2)],
    ["Kjønn", "/".join(np.unique(sex_treatment, return_counts=True)[1].astype(str)), "/".join(np.unique(sex_control, return_counts=True)[1].astype(str)), round(smd_sex, 2)],
    ["SBP (mmHg)", round(np.mean(sbp_treatment), 2), round(np.mean(sbp_control), 2), round(smd_sbp, 2)],
    ["Røyking", "/".join(np.unique(smoking_treatment, return_counts=True)[1].astype(str)), "/".join(np.unique(smoking_control, return_counts=True)[1].astype(str)), round(smd_smoking, 2)],
    ["Trening", "/".join(np.unique(exercise_treatment, return_counts=True)[1].astype(str)), "/".join(np.unique(exercise_control, return_counts=True)[1].astype(str)), round(smd_exercise, 2)]
]

table = tabulate(data, headers='firstrow', tablefmt='pipe')
print(table)





