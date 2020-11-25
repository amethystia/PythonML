#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/26 15:21
# @Author  : louwill
# @File    : lasso.py
# @mail: ygnjd2016@gmail.com


import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


class Lasso():
    def __init__(self):
        pass

    def prepare_data(self):
        data = np.genfromtxt('F:/machine-learning-code-writing-master/lasso/example.dat', delimiter=',')
        # 选择特征和标签
        x = data[:, 0:100]
        y = data[:, 100].reshape(-1, 1)
        # 加一列
        X = np.column_stack((np.ones((x.shape[0], 1)), x))
        # 划分训练集和测试集
        X_train, y_train = X[:70], y[:70]
        X_test, y_test = X[70:], y[70:]
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
        return X_train, y_train, X_test, y_test

    # 定义初始化函数
    def initialize_params(self, dims):
        w = np.zeros((dims, 1))
        b = 0
        return w, b

    # 定义符号函数
    def sign(self, x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0

    # vec_sign = np.vectorize(sign)
    # vec_sign(np.zeros((3, 1)))

    #

    def l1_loss(self, X, y, w, b, alpha):
        num_train = X.shape[0]
        num_feature = X.shape[1]

        y_hat = np.dot(X, w) + b
        loss = np.sum((y_hat - y) ** 2) / num_train + np.sum(alpha * abs(w))
        dw = np.dot(X.T, (y_hat - y)) / num_train + alpha * np.vectorize(self.sign)(w)
        db = np.sum((y_hat - y)) / num_train
        return y_hat, loss, dw, db

    def lasso_train(self, X, y, learning_rate, epochs):
        loss_list = []
        w, b = self.initialize_params(X.shape[1])
        for i in range(1, epochs):
            y_hat, loss, dw, db = self.l1_loss(X, y, w, b, 0.1)
            w += -learning_rate * dw
            b += -learning_rate * db
            loss_list.append(loss)

            if i % 50 == 0:
                print('epoch %d loss %f' % (i, loss))

            params = {
                'w': w,
                'b': b
            }
            grads = {
                'dw': dw,
                'db': db
            }
        return loss, loss_list, params, grads

    def predict(self, X, params):
        w = params['w']
        b = params['b']
        y_pred = np.dot(X, w) + b
        return y_pred


if __name__ == '__main__':
    lasso = Lasso()
    X_train, y_train, X_test, y_test = lasso.prepare_data()
    loss, loss_list, params, grads = lasso.lasso_train(X_train, y_train, 0.01, 500)
    print(params)
    y_pred = lasso.predict(X_test, params)
    print(r2_score(y_test, y_pred))
    # 简单绘图

    f = X_test.dot(params['w']) + params['b']

    plt.scatter(range(X_test.shape[0]), y_test)
    plt.plot(f, color='darkorange')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show();

    # 训练过程中的损失下降
    plt.plot(loss_list, color='blue')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()
    # 导入线性模型模块
    from sklearn import linear_model

    # 创建lasso模型实例
    sk_lasso = linear_model.Lasso(alpha=0.1)
    # 对训练集进行拟合
    sk_lasso.fit(X_train, y_train)
    # 打印模型相关系数
    print("sklearn Lasso intercept :", sk_lasso.intercept_)
    print("\nsklearn Lasso coefficients :\n", sk_lasso.coef_)
    print("\nsklearn Lasso number of iterations :", sk_lasso.n_iter_)

    # %%
