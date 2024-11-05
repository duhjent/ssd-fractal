import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('./outs/run-Sep15_19-46-04_606d7b92c104SSD-DFG-tag-Loss_train.csv')

print(df.Step.min(), df.Step.max())
