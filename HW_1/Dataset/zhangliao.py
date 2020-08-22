# import csv
import numpy as np
import pandas as pd

weight_vector = np.random.rand(1, 153).flatten()

bias = 0.0001
# dimension_number = 0


def init_today_data():
    return np.empty((18, 24))


def remove_NR(value):
    if value == "NR":
        return 0
    else:
        return value


def new_rainfall(origin_row):
    # for index, row in origin_row.iterrows():
    new_row = origin_row.map(remove_NR)
    print(new_row)
    print(type(new_row))
    return new_row
    """
    rainfall = np.empty(24)
    for i in range(len(origin_row)):
        if origin_row[i] != "NR":
            rainfall[i] = origin_row[i]
        else:
            rainfall[i] = 0  # 0??? nan
    return rainfall
   """


def compute_loss_and_grad(features, real_data, today_loss, today_grad_w, today_grad_b):
    predict_data = np.dot(features, weight_vector) + bias
    loss = np.square(predict_data - real_data)
    grad_w = 2 * (real_data - predict_data) * (-features)
    grad_b = -2 * (real_data - predict_data)
    return today_loss + loss, today_grad_w + grad_w, today_grad_b + grad_b


def one_day_loss_and_grad(data, loss, grad_w, grad_b):
    today_loss = 0
    today_grad_w = 0
    today_grad_b = 0
    today_pm25 = data[9]
    # print("data")
    # print(data)
    data_without_pm = np.delete(data, 9, 0)
    # print("data_without_pm")
    # print(data_without_pm)
    # print(data_without_pm)
    features1 = (data_without_pm[:, 0:9]).flatten()
    # print(compute_loss(features1, today_pm25[9]))
    today_loss, today_grad_w, today_grad_b = compute_loss_and_grad(
        features1, today_pm25[9], today_loss, today_grad_w, today_grad_b
    )
    features2 = data_without_pm[:, 10:19].flatten()
    today_loss, today_grad_w, today_grad_b = compute_loss_and_grad(
        features2, today_pm25[19], today_loss, today_grad_w, today_grad_b
    )
    return loss + today_loss, grad_w + today_grad_w, grad_b + today_grad_b


"""
def get_row_data(row):
    return list(row.values())[3:]
"""


def get_row_data(row):
    return row[3:]


"""
def train_one_loop(weight_vector, bias):

    with open("train.csv") as f:
        f_csv = csv.DictReader(f)
        # print(f_csv)
        today_data = init_today_data()
        loss = 0
        grad_w = np.zeros(153)
        grad_b = 0
        i = 0
        for row in f_csv:
            # print(list(row.values())[3:])
            if i == 10:
                today_data[i] = rainfall(get_row_data(row))
            # elif i == 9:
            #   today_pm25 = get_row_data(row)
            else:
                today_data[i] = get_row_data(row)

            i += 1

            if i == 18:                
                # loss = loss +
                loss, grad_w, grad_b = one_day_loss_and_grad(
                    today_data, loss, grad_w, grad_b
                )
                # print("*************")
                # print(today_data)
                i = 0
                today_data = init_today_data()

    return loss, grad_w, grad_b
"""

"""
def train_one_loop(weight_vector, bias, df):

    today_data = init_today_data()
    loss = 0
    grad_w = np.zeros(153)
    grad_b = 0
    i = 0
    for index, row in df.iterrows():
        if i == 10:
            today_data[i] = rainfall(get_row_data(row))
        else:
            today_data[i] = get_row_data(row)
        i += 1

        if i == 18:
            loss, grad_w, grad_b = one_day_loss_and_grad(
                today_data, loss, grad_w, grad_b
            )
            i = 0
            today_data = init_today_data()

    return loss, grad_w, grad_b
"""

"""
def eval (weight_vector, bias)
    test_data = pd.read_csv("train.csv")
"""


def train_one_loop(weight_vector, bias, train_data):
    day_number = 0
    loss = 0
    grad_w = np.zeros(153)
    grad_b = 0
    # print(train_data.iloc[0:18])

    for day_number in range(240):
        today_index = day_number * 18
        one_day_data = train_data.iloc[today_index : today_index + 18]
        # one_day_data = train_data.iloc[today_index : today_index + 18]
        # print(one_day_data["1"])
        loss, grad_w, grad_b = one_day_loss_and_grad(
            one_day_data.to_numpy(dtype="float"), loss, grad_w, grad_b
        )
    return loss, grad_w, grad_b


step = 0.000000001
loss = 10000  # choose a number bigger than 1000 so that code in while works
df = pd.read_csv("train.csv")

data_without_useless_info = df.drop(["Date", "stations", "observation"], axis=1)
train_data = data_without_useless_info.replace(["NR"], 0)

while loss > 1000:
    loss, grad_w, grad_b = train_one_loop(weight_vector, bias, train_data)
    weight_vector = weight_vector - step * grad_w
    bias = bias - step * grad_b
    print("loss")
    print(loss)

"""
print("weight vector")
print(weight_vector)
print("bias")
print(bias)
"""
