import numpy as np
import matplotlib.pyplot as plt

# 准备训练数据
from numpy import random

data_dict = {-1: np.array([[1, 7],
                           [2, 8],
                           [3, 8], ]),
             1: np.array([[5, 1],
                          [6, -1],
                          [7, 3], ])}
print(data_dict)


# 定义线性可分支持向量机的模型主体和训练部分
def train(data):
    # 参数字典 { ||w||: [w,b] }
    opt_dict = {}

    # 数据转换列表
    transforms = [[1, 1],
                  [-1, 1],
                  [-1, -1],
                  [1, -1]]

    # 从字典中获取所有数据
    all_data = []
    for yi in data:
        for featureset in data[yi]:
            for feature in featureset:
                all_data.append(feature)

    # 获取数据最大最小值
    max_feature_value = max(all_data)
    min_feature_value = min(all_data)
    all_data = None

    # 定义一个学习率(步长)列表
    step_sizes = [max_feature_value * 0.1,
                  max_feature_value * 0.01,
                  max_feature_value * 0.001
                  ]

    # 参数b的范围设置
    b_range_multiple = 2
    b_multiple = 5
    latest_optimum = max_feature_value * 10

    # 基于不同步长训练优化
    for step in step_sizes:
        w = np.array([latest_optimum, latest_optimum])
        # 凸优化
        optimized = False
        while not optimized:
            for b in np.arange(-1 * (max_feature_value * b_range_multiple),
                               max_feature_value * b_range_multiple,
                               step * b_multiple):
                for transformation in transforms:
                    w_t = w * transformation
                    found_option = True

                    for i in data:
                        for xi in data[i]:
                            yi = i
                            if not yi * (np.dot(w_t, xi) + b) >= 1:
                                found_option = False
                                # print(xi,':',yi*(np.dot(w_t,xi)+b))

                    if found_option:
                        opt_dict[np.linalg.norm(w_t)] = [w_t, b]

            if w[0] < 0:
                optimized = True
                print('Optimized a step!')
            else:
                w = w - step

        norms = sorted([n for n in opt_dict])
        # ||w|| : [w,b]
        opt_choice = opt_dict[norms[0]]
        w = opt_choice[0]
        b = opt_choice[1]
        latest_optimum = opt_choice[0][0] + step * 2

    for i in data:
        for xi in data[i]:
            yi = i
            print(xi, ':', yi * (np.dot(w, xi) + b))
    return w, b, min_feature_value, max_feature_value


def hyperplane(x, w, b, v):
    return (-w[0] * x - b + v) / w[1]


w, b, min_feature_value, max_feature_value = train(data_dict)

# 可视化
colors = {1: 'r', -1: 'g'}
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

import matplotlib.markers


# 定义预测函数
def predict(features):
    # sign( x.w+b )
    classification = np.sign(np.dot(np.array(features), w) + b)
    if classification != 0:
        ax.scatter(features[0], features[1], s=50, marker='s', c=colors[classification])
        print(classification)
    return classification


# 从给定上下限范围选取随机整数，生成数组
# randint(最小值, 最大值, (size))

predict_us = random.randint(1, 8, (20, 2))
print(predict_us)
for p in predict_us:
    predict(p)

[[ax.scatter(x[0], x[1], s=100, color=colors[i]) for x in data_dict[i]] for i in data_dict]

# hyperplane = x.w+b
# v = x.w+b
# psv = 1
# nsv = -1
# dec = 0
# 定义线性超平面


datarange = (min_feature_value * 0.9, max_feature_value * 1.1)
hyp_x_min = datarange[0]
hyp_x_max = datarange[1]

# (w.x+b) = 1
# 正支持向量
psv1 = hyperplane(hyp_x_min, w, b, 1)
psv2 = hyperplane(hyp_x_max, w, b, 1)
ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')

# (w.x+b) = -1
# 负支持向量
nsv1 = hyperplane(hyp_x_min, w, b, -1)
nsv2 = hyperplane(hyp_x_max, w, b, -1)
ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

# (w.x+b) = 0
# 线性分隔超平面
db1 = hyperplane(hyp_x_min, w, b, 0)
db2 = hyperplane(hyp_x_max, w, b, 0)
ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

plt.show()
