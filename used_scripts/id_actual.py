import pandas as pd
import numpy as np

df = pd.read_csv('/home/martianspeaks/Study/FINAL.csv')
x1 = np.array(df['x1'])
x2 = np.array(df['x2'])

dict = {}
for A, B in zip(x1, x2):
    dict[A] = B

j = 0
for i in df['category_id']:
    df['class'][j] = dict[i]
    j += 1


print (df['class'])

df.to_csv('FINAL.csv',index=False)
