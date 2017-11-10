import pandas as pd

albrecht = pd.read_excel('Albrecht.xlsx')
df = pd.DataFrame(albrecht)
print df

"""max value
input_max = df['Input'].max()
print input_max"""

""" min value
input_min = df['Input'].min()
print input_min"""


""" get item by row,column
items = df.at[4, 'Input']
print items"""





