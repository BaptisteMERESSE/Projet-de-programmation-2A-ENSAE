import pandas as pd
import numpy as np

a = np.array([1,2,3])

data = {
    'A': [1, 1, 1],
    'B': [7, 3, 4],
    'C': [2, 8, 9],
    'D': [10, 7, 10]
}
data = pd.DataFrame(data)

def mean(data, column_name):
    m = data[column_name].mean()
    return m

def type_free_mean(data, column_name):
    first_element = True
    count = 0
    for i in data[column_name]:

        count += 1
        if first_element:
            M = i
            first_element = False
        else:
            M += i
    m = M* (1/count)
    return m


def var(data, column_name):
    working_data = data[[column_name]].copy()
    working_data["X2"] = working_data[column_name]*working_data[column_name]
    v = mean(working_data, "X2") - mean(working_data, column_name)**2
    return v

def cov(data, column_name_1, column_name_2):
    working_data = data[[column_name_1, column_name_2]]
    working_data["Mean 1"] = mean(data, column_name_1)
    working_data["Mean 2"] = mean(data, column_name_2)
    working_data["Cov"] = (working_data[column_name_1] - working_data["Mean 1"])*(working_data[column_name_2] - working_data["Mean 2"])
    cov = working_data["Cov"].mean()
    return cov
    
def cor(data, column_name_1, column_name_2):
    covar = cov(data, column_name_1, column_name_2)
    var_1 = var(data, column_name_1)
    var_2 = var(data, column_name_2)
    e1 = var_1**(1/2)
    e2 = var_2**(1/2)
    c = covar/(e1*e2)
    return c

def mse_regression(data, column_list, interest_var):
    data = data
    k = len(column_list)
    n = len(data[interest_var])

    work_data = data[column_list + [interest_var]].copy()
    work_data["Vectorized"] = work_data.apply(lambda row: [[row[column_name] for column_name in column_list]], axis = 1)
    work_data["VectorizedT"] =  work_data.apply(lambda row: np.transpose([[row[column_name] for column_name in column_list]]), axis = 1)
    work_data["XX^T"] = work_data.apply(lambda row: np.dot(row["VectorizedT"], row["Vectorized"]), axis = 1)
    mean_1 = type_free_mean(work_data, "XX^T")

    work_data["XY"] = work_data.apply(lambda row: (np.transpose([[row[column_name] for column_name in column_list]]))* row[interest_var], axis = 1)
    mean_2 = type_free_mean(work_data, "XY")

    inv_mean_1 = np.linalg.inv(mean_1)

    beta = np.dot(inv_mean_1, mean_2)
    data["Regression"] = data.apply(lambda row: sum([row[column_list[i]]*beta[i] for i in range(len(column_list))])[0], axis = 1)
    R2 = var(data, "Regression")/var(data, interest_var)
    R2_adj = 1 - (1-R2)*(n-1)/(n-k)
    data["Regression"] = data["Regression"].round(2)

    return beta, R2, R2_adj




    

""" 
d = [data["A"].values]
print(np.transpose(d))
print(np.dot(np.transpose(d), d))
"""
#print(type_free_mean(data, "A"))
#print(mse_regression(data, ["A", "B"], "D"))
#print(data.head())