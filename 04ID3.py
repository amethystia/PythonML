import io

import numpy as np
import pandas as pd
from math import log


# 定义熵计算函数
def entropy(ele):
    probs = [ele.count(i) / len(ele) for i in set(ele)]
    entropy = - sum([prob * log(prob, 2) for prob in probs])  # =-
    return entropy


# 定义根据特征和特征值进行数据划分的方法
def split_dataframe(data, col):
    # 每列的特征值
    eigenvalue = data[col].unique()
    #
    result_dict = {elem: pd.DataFrame for elem in eigenvalue}
    # 基于列值划分数据
    for key in result_dict.keys():
        result_dict[key] = data[:][data[col] == key]
    return result_dict


# 根据熵计算公式和数据集划分方法计算信息增益来选择最佳特征
def choose_best_col(df, label):
    # 计算标签的熵
    entropy_D = entropy(df[label].tolist())
    # columns list except label
    cols = [col for col in df.columns if col not in [label]]
    max_value, best_col = -999, None
    max_splited = None
    # split data based on different column
    for col in cols:
        splited_set = split_dataframe(df, col)
        entropy_DA = 0
        for subset_col, subset in splited_set.items():
            # calculating splited dataframe label's entropy
            entropy_Di = entropy(subset[label].tolist())
            # calculating entropy of current feature
            entropy_DA += len(subset) / len(df) * entropy_Di
        # calculating infomation gain of current feature
        info_gain = entropy_D - entropy_DA

        if info_gain > max_value:
            max_value, best_col = info_gain, col
            max_splited = splited_set
    return max_value, best_col, max_splited


class ID3Tree:
    class Node:
        def __init__(self, name):
            self.name = name
            self.connections = {}

        def connect(self, label, node):
            self.connections[label] = node

    def __init__(self, data, label):
        self.columns = data.columns
        self.data = data
        self.label = label
        self.root = self.Node("Root")

    def print_tree(self, node, tabs):
        # 不加str强转会报错 TypeError: can only concatenate str (not "numpy.bool_") to str
        print(tabs + str(node.name))
        for connection, child_node in node.connections.items():
            print(tabs + "\t" + "(" + str(connection) + ")")
            self.print_tree(child_node, tabs + "\t\t")

    def construct_tree(self):
        self.construct(self.root, "", self.data, self.columns)

    def construct(self, parent_node, parent_connection_label, input_data, columns):
        max_value, best_col, max_splited = choose_best_col(input_data[columns], self.label)

        if not best_col:
            node = self.Node(input_data[self.label].iloc[0])
            parent_node.connect(parent_connection_label, node)
            return

        node = self.Node(best_col)
        parent_node.connect(parent_connection_label, node)

        new_columns = [col for col in columns if col != best_col]

        for splited_value, splited_data in max_splited.items():
            self.construct(node, splited_value, splited_data, new_columns)


# dtype
df = pd.read_csv('F:/machine-learning-code-writing-master/id3/mytest.csv')
print(df)

print(entropy(df['eat'].tolist()))
print('        ')

split_example = split_dataframe(df, 'age')
print(split_example)

print(choose_best_col(df, 'age'))

tree1 = ID3Tree(df, 'eat')
tree1.construct_tree()
tree1.print_tree(tree1.root, "")

# sklearn tree模块调用决策树
from sklearn.datasets import load_iris
from sklearn import tree
import graphviz

iris = load_iris()

# criterion : {"gini", "entropy"}, default="gini"
# splitter : {"best", "random"}, default="best"

clf = tree.DecisionTreeClassifier(criterion='entropy', splitter='best')
# build a decision tree classifier from the training set (X, y).    模型训练
clf = clf.fit(iris.data, iris.target)

# This function generates a GraphViz representation of the decision tree,
#     which is then written into `out_file`
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=iris.feature_names,
                                class_names=iris.target_names, filled=True, rounded=True, special_characters=True,
                                precision=10)
# Verbatim DOT source code string to be rendered by Graphviz
graph = graphviz.Source(dot_data, filename='ID3test', format='png')
# Save the source to file, open the rendered result in a viewer.
graph.view()

# example
# import graphviz from Digraph

# dot = Digraph(comment='The Round Table')
# dot.node('A', 'King Arthur')
# dot.node('B', 'Sir Bedevere the Wise')
# dot.node('L', 'Sir Lancelot the Brave')
# dot.edges(['AB', 'AL'])
# dot.edge('B', 'L', constraint='false')
# print(dot.source)
