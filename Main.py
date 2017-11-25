import pandas as pd
import math as m
import numpy as np

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 113)
# end import

albrecht = pd.read_excel('Albrecht.xlsx')
df = pd.DataFrame(albrecht)

print ('---------------------------------INPUT-------------------------------------------')
print df
print ('---------------------------------------------------------------------------------')

# create zero matrix
dist = np.zeros([df.shape[0], df.shape[0]])


def get_dist():  # calculate distance func
    for i in range(0, len(df)):
        for j in range(0, len(df)):
            if i != j:
                cal_dist = m.sqrt(
                    ((df.at[i, 'Input'] - df.at[j, 'Input']) ** 2) + ((df.at[i, 'output'] - df.at[j, 'output']) ** 2)
                    + ((df.at[i, 'Inquiry'] - df.at[j, 'Inquiry']) ** 2) + ((df.at[i, 'File'] - df.at[j, 'File']) ** 2)
                    + ((df.at[i, 'FPAdj'] - df.at[j, 'FPAdj']) ** 2) + ((df.at[i, 'RawFP'] - df.at[j, 'RawFP']) ** 2)
                    + ((df.at[i, 'AdjFP'] - df.at[j, 'AdjFP']) ** 2))
                result = round(cal_dist, 2)
                dist[i][j] = result


get_dist()  # call function

# print dist  # matrix result
# create data frame for distance result
print ('--------------------------------------------DISTANCE '
       'TABLE-------------------------------------------------------') 
dist_df = pd.DataFrame(dist)
print dist_df
print ('---------------------------------------------------------'
       '-------------------------------------------------------')

print ('-----------------------------------SORTED DISTANCE '
       'TABLE-------------------------------------------------------')
inf = float('inf')
np.fill_diagonal(dist, inf)

s_d = np.sort(dist)
dist_sorted = pd.DataFrame(s_d)
print dist_sorted
print ('---------------------------------------------------------'
       '-------------------------------------------------------')

print ("-------------------------INDEX OF SIMILAR CASE------"
       "--------------------------------")

# index ranking
index_val = np.argsort(dist)
print index_val

ranking = []
for i in range(len(index_val)):
    row = []
    for j in range(3):
            row.append(index_val[i, j])
    ranking.append(row)

# data value based on ranking
dist_val = []
for i in range(len(s_d)):
    row = []
    for j in range(3):
        row.append(float(s_d[i, j]))
    dist_val.append(row)

#Effort value based on ranking
effort_val = []
for i in range(len(index_val)):
    row = []
    for j in range(3):
        row.append(float(df.at[index_val[i, j], 'Effort']))
    effort_val.append(row)


size_val = []
for i in range(len(index_val)):
    row = []
    for j in range(3):
        row.append(df.at[index_val[i, j], 'AdjFP'])
    size_val.append(row)


# Ranking df
df_ranking = pd.DataFrame(ranking)
df_dist_val = pd.DataFrame(dist_val)
df_effort = pd.DataFrame(effort_val)
df_size = pd.DataFrame(size_val)
print ('---------------RANKING INDEX-----------------')
print df_ranking
print ('---------------RANKING DISTANCE-----------------')
print df_dist_val
print ('---------------RANKING EFFORT-----------------')
print df_effort
print ('---------------RANKING SIZE-----------------')
print df_size

# UAVG Calculation
UAVG = []
for i in range(len(df_effort)):
    row = []
    row.append(float((df_effort.at[i, 0]+df_effort.at[i, 1]+df_effort.at[i, 2])/3))
    UAVG.append(row)

#IRWM Calculation
IRWM = []
for i in range(len(df_effort)):
    row = []
    result = 0
    for j in range(3):
        result += ((3 - (j+1) + 1) * df_effort.at[i, j])
    row.append(result/6)
    IRWM.append(row)




print ('---------------END MAPPING-----------------')

print ('---------------UAVG CALCULATION-----------------')
df_UAVG = pd.DataFrame(UAVG)
print df_UAVG

print ('---------------IRWM CALCULATION-----------------')
df_IRWM = pd.DataFrame(IRWM)
print df_IRWM


# LSA Calculation
LSA = []
for i in range(len(df_size)):
    row = []
    size_new = df.at[i, 'AdjFP']
    size_analogue = df_size.at[i, 0] + df_size.at[i, 1] + df_size.at[i, 2]
    result = size_new / (size_analogue/3.0)
    row.append(result * df_UAVG.at[i, 0])
    LSA.append(row)


print ('---------------LSA CALCULATION-----------------')
df_LSA = pd.DataFrame(LSA)
print df_LSA

print ('-----------------------------------------------')

# Cal mean

UAVG_mean = 0
sum_uavg = 0
for i in range(len(df_UAVG)):
    sum_uavg += m.fabs(df_UAVG.at[i, 0] - df.at[i, 'Effort'])
UAVG_mean = round(sum_uavg/24.0, 2)

IRWM_mean = 0
sum_irwm = 0
for i in range(len(df_IRWM)):
    sum_irwm += m.fabs(df_IRWM.at[i, 0] - df.at[i, 'Effort'])
IRWM_mean = round(sum_irwm/24.0, 2)

LSA_mean = 0
sum_lsa = 0
for i in range(len(df_LSA)):
    sum_lsa += m.fabs(df_LSA.at[i, 0] - df.at[i, 'Effort'])
LSA_mean = round(sum_lsa/24.0, 2)

print ('----Accuracy calculation----')

print ('UAVG mean is : ' + str(UAVG_mean))
print ('IRWM mean is : ' + str(IRWM_mean))
print ('LSA mean is  : ' + str(LSA_mean))




