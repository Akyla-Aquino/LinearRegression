import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
# Load the data
oecd_bli = pd.read_csv("oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv("gdp_per_capita.csv", thousands=',', delimiter='\t',
                             encoding='latin1', na_values="n/a")
# Prepare the data


def prepareData(oecd_bli, gdp_per_capita):

    oecd_bli = oecd_bli[oecd_bli['INEQUALITY']
                        == 'TOT']  # Selecionando dados TOT
    oecd_bli = oecd_bli.pivot(
        index="Country", columns="Indicator", values="Value")  # Reajustando dados

    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)

    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)

    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]


country_stats = prepareData(oecd_bli, gdp_per_capita)

x = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

plt.plot(x, y,'*')

b = ((np.sum(x*y)) - (np.mean(y) * np.sum(x)))/((np.sum(x**2)) - (np.sum(x)**2/2250))
a = np.mean(y) - b*np.mean(x)
y_prevision = a + b*x 

plt.plot(x, y_prevision)
plt.show()
