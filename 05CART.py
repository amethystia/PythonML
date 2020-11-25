import numpy as np
import pandas as pd

lst = ['a', 'b', 'c', 'd', 'b', 'c', 'a', 'b', 'c', 'd']


def gini(nums):
    probs = [nums.count(i) / len(nums) for i in set(nums)]
    print(probs)
    gini = sum([p * (1 - p) for p in probs])
    return gini


print(gini(lst))

df = pd.read_csv('F:/machine-learning-code-writing-master/id3/example_data.csv',
                 dtype={'windy': 'str'})
dfl = df['play'].tolist()
print(dfl)
gini(dfl)


# 定义根据特征分割数据框的函数
def split_dataframe(data, col):
    '''
    function: split pandas dataframe to sub-df based on data and column.
    input: dataframe, column name.
    output: a dict of splited dataframe.
    '''
    # unique value of column
    unique_values = data[col].unique()
    # empty dict of dataframe
    result_dict = {elem: pd.DataFrame for elem in unique_values}
    # split dataframe based on column value
    for key in result_dict.keys():
        result_dict[key] = data[:][data[col] == key]
    return result_dict


print(split_dataframe(df, 'humility'))


# 根据Gin指数和条件Gini指数计算递归选择最优特征

def choose_best_col(df, label):
    '''
    funtion: choose the best column based on infomation gain.
    input: datafram, label
    output: max infomation gain, best column,
            splited dataframe dict based on best column.
    '''
    # Calculating label's gini index
    gini_D = gini(df[label].tolist())
    # columns list except label
    cols = [col for col in df.columns if col not in [label]]
    # initialize the max infomation gain, best column and best splited dict
    min_value, best_col = 999, None
    min_splited = None
    # split data based on different column
    for col in cols:
        splited_set = split_dataframe(df, col)
        gini_DA = 0
        for subset_col, subset in splited_set.items():
            # calculating splited dataframe label's gini index
            gini_Di = gini(subset[label].tolist())
            # calculating gini index of current feature
            gini_DA += len(subset) / len(df) * gini_Di

        if gini_DA < min_value:
            min_value, best_col = gini_DA, col
            min_splited = splited_set
    return min_value, best_col, min_splited


print(choose_best_col(df, 'play'))


class CartTree:
    # 定义结点类
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

    # 打印树方法
    def print_tree(self, node, tabs):
        print(tabs + node.name)
        for connection, child_node in node.connections.items():
            print(tabs + "\t" + "(" + connection + ")")
            self.print_tree(child_node, tabs + "\t\t")

    def construct_tree(self):
        self.construct(self.root, "", self.data, self.columns)

        # 构造树

    def construct(self, parent_node, parent_connection_label, input_data, columns):
        min_value, best_col, min_splited = choose_best_col(input_data[columns], self.label)
        if not best_col:
            node = self.Node(input_data[self.label].iloc[0])
            parent_node.connect(parent_connection_label, node)
            return

        node = self.Node(best_col)
        parent_node.connect(parent_connection_label, node)

        new_columns = [col for col in columns if col != best_col]
        # 递归构造决策树
        for splited_value, splited_data in min_splited.items():
            self.construct(node, splited_value, splited_data, new_columns)


tree1 = CartTree(df, 'play')
tree1.construct_tree()
tree1.print_tree(tree1.root, "")

from sklearn.datasets import load_iris
from sklearn import tree
import graphviz

iris = load_iris()

# criterion : {"gini", "entropy"}, default="gini"
# splitter : {"best", "random"}, default="best"

clf = tree.DecisionTreeClassifier(criterion='gini', splitter='best')
# build a decision tree classifier from the training set (X, y).    模型训练
clf = clf.fit(iris.data, iris.target)

# This function generates a GraphViz representation of the decision tree,
#     which is then written into `out_file`
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=iris.feature_names,
                                class_names=iris.target_names, filled=True, rounded=True, special_characters=True,
                                precision=10)
# Verbatim DOT source code string to be rendered by Graphviz
graph = graphviz.Source(dot_data, filename='CARTtest', format='png')
# Save the source to file, open the rendered result in a viewer.
graph.view()

