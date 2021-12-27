import pandas as pd
from dataclasses import make_dataclass

Ped_sample = make_dataclass("Sample", [("Actions", float), ("Ground_Truth", float), ("Image_Name", str)])

df = pd.read_csv('Herdr_data.csv')

print(df)



