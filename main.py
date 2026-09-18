from Data.pegel_utils import load_data, plot_pegel_data
import pandas as pd

#Läd Pegeldaten von bestimmtem Zeitpunkt
pegel_data = load_data('2022-01-01', '2022-05-04')

print(pegel_data.head())
print(pegel_data.describe())

#Plotet daten ab gewähltem Datum
#optional kann ein plot_to Datum mitgegeben werden, bis wann die Daten geplottet werden sollen
plot_from = '2022-01-01'
plot_pegel_data(pegel_data, plot_from)