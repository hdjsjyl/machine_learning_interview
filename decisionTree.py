import pickle as pc
import numpy as np
import pandas as pd
import random
random.seed(100)

## decision tree ID3: 信息增益
def information_entropy(counts):
    percentages = counts/np.sum(counts)
    s = 0
    for per in percentages:
        if per == 0:
            continue
        s += -1*(per*np.log2(per))
    return s

## gini 指数
def gini_impurity(counts):
    p_list=counts/np.sum(counts)
    return 1-np.sum(p_list*p_list)

def data_entropy(data):
    n_rows = data.shape[0]
    n_low = data[data.low==1].shape[0]
    n_high = n_rows - n_low
    return information_entropy([n_low, n_high])

def data_entropy2(data1, data2):
    entropy1 = data_entropy(data1)
    entropy2 = data_entropy(data2)
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    n = n1 + n2
    return n1/n*entropy1 + n2/n*entropy2

def find_best_feature(data, label):
    x = data.drop(label, axis=1)
    min_entropy = 1
    col_selected = ''
    data_positive_found = None
    data_negative_found = None
    for col in x.columns:
        data_positive = data[data[col] == 1]
        data_negative = data[data[col] == 0]
        if data_positive.shape[0] == 0 or data_negative.shape[0] == 0:
            continue
        entropy = data_entropy2(data_positive, data_negative)
        if entropy < min_entropy:
            min_entropy = entropy
            col_selected = col
            data_positive_found = data_positive
            data_negative_found = data_negative_found

    return col_selected, min_entropy, data_positive_found, data_negative_found

class Branch:
    no = 0
    depth = 1
    column = ''
    entropy = 0
    samples = 0
    value = []

    branch_positive = None
    branch_negative = None
    no_positive = 0
    no_negative = 0

number = 0
def decision_tree_inner(data, label, depth, max_depth=3):
    print(depth)
    global number
    branch = Branch()
    branch.no = number
    number    = number + 1
    branch.depth = depth
    branch.samples = data.shape[0]
    n_positive = data[data[label] == 1].shape[0]
    branch.value = [branch.samples-n_positive, n_positive]
    branch.entropy = information_entropy(branch.value)
    best_features = find_best_feature(data, label)
    branch.column = best_features[0]
    # print(branch.column)
    new_entropy   = best_features[1]
    if depth == max_depth or branch.column == '':
        branch.no_positive = number
        number += 1
        branch.no_negative = number
        number += 1
        return branch
    else:
        data_negative = best_features[3]
        if data_negative is not None:
            branch.branch_negative = decision_tree_inner(data_negative, label, depth+1, max_depth)

        data_positive = best_features[2]
        if data_positive is not None:
            branch.brach_positive = decision_tree_inner(data_positive, label, depth+1, max_depth)

        return branch

def decision_tree(data, label, max_depth=3):
    number = 0
    entropy = data_entropy(data)
    tree = decision_tree_inner(data, label, 0, max_depth=3)
    return tree


def get_dot_data_innner(branch: Branch, classes, dot_data):
    if branch.value[0] < branch.value[1]:
        the_class = classes[0]
    else:
        the_class = classes[1]
    if branch.branch_positive:
        dot_data = dot_data + '{} [label=<{}?<br/>entropy = {:.3f}<br/>samples = {}<br/>value = {}<br/>class = {}> , fillcolor="#FFFFFFFF"] ;\r\n'.format(
            branch.no, branch.column, branch.entropy, branch.samples, branch.value, the_class)
    else:
        dot_data = dot_data + '{} [label=<entropy = {:.3f}<br/>samples = {}<br/>value = {}<br/>class = {}> , fillcolor="#FFFFFFFF"] ;\r\n'.format(
            branch.no, branch.entropy, branch.samples, branch.value, the_class)
    if branch.branch_negative:
        dot_data = dot_data + '{} -> {} [labeldistance=2.5, labelangle=45, headlabel="no"]; \r\n'.format(branch.no,
                                                                                                         branch.branch_negative.no)
        dot_data = get_dot_data_innner(branch.branch_negative, classes, dot_data)

    if branch.branch_positive:
        dot_data = dot_data + '{} -> {} [labeldistance=2.5, labelangle=45, headlabel="yes"]; \r\n'.format(branch.no,
                                                                                                          branch.branch_positive.no)
        dot_data = get_dot_data_innner(branch.branch_positive, classes, dot_data)

    return dot_data


def get_dot_data(branch: Branch, classes=['low', 'high']):
    dot_data = """
digraph Tree {
node [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;
edge [fontname=helvetica] ;
"""
    dot_data = get_dot_data_innner(branch, classes, dot_data)
    dot_data = dot_data + '\r\n}'
    return dot_data

data = pc.load(open('./shanghai_experience_3_5.dmp', mode='rb'))
mask = np.random.rand(len(data)) < 0.8
data_train = data[mask]
data_test = data[~mask]
print(data_train.shape[0])
# x_train = data_train.drop('low', axis=1)
# y_train = data_train.low

# samples = x_train.shape[0]

my_dt = decision_tree(data_train, 'low', max_depth=3)
branch = my_dt
print(branch.column)
# left = branch.branch_negative
# if left:
#     print(left.column)
right = branch.branch_positive
# if right:
print(right.column)


dot_data=get_dot_data(my_dt)

import graphviz
graph = graphviz.Source(dot_data)
graph.render('./data/my_dt', format='png')
graph