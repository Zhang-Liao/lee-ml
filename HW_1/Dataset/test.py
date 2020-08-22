import numpy as np
import pandas as pd

weight_vector = np.random.rand(1, 153).flatten()

bias = 0.0001

df = pd.read_csv("test.csv", names=["category", 0, 1, 2, 3, 4, 5, 6, 7, 8])

data_without_useless_info = df.drop(["category"], axis=1)
test_data = data_without_useless_info.replace(["NR"], 0)

# print(test_data)


def compute_loss(one_day_data, real_data):
    data_without_pm = np.delete(one_day_data, 9, 0)
    predict_data = np.dot(features, weight_vector) + bias
    loss = abs(predict_data - real_data)
    return loss


# def predict (weight, bias, features):

for day_number in range(120):
    today_index = day_number * 18
    one_day_data = test.iloc[today_index : today_index + 18]
    today_pm25 = data[9]

