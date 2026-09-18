from Data.access_data import load_data
import pandas as pd


pegel_data = load_data('2022-01-01')

print(pegel_data.head())
print(pegel_data.describe())
