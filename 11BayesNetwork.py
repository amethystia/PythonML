# from pgmpy.factors.discrete import TabularCPD
# from pgmpy.models import BayesianModel
#
# # 指定模型之间的依赖关系conda
# student_model = BayesianModel([('D', 'G'),
#                                ('I', 'G'),
#                                ('G', 'L'),
#                                ('I', 'S')])
# # 构造各个节点和传入概率表并指定相关参数
# grade_cpd = TabularCPD(
#     variable='G',  # 节点名称
#     variable_card=3,  # 节点取值个数
#     values=[[0.3, 0.05, 0.9, 0.5],  # 该节点的概率表
#             [0.4, 0.25, 0.08, 0.3],
#             [0.3, 0.7, 0.02, 0.2]],
#     evidence=['I', 'D'],  # 该节点的依赖节点
#     evidence_card=[2, 2]  # 依赖节点的取值个数
# )
# difficulty_cpd = TabularCPD(
#     variable='D',
#     variable_card=2,
#     values=[[0.6, 0.4]]
# )
#
# intel_cpd = TabularCPD(
#     variable='I',
#     variable_card=2,
#     values=[[0.7, 0.3]]
# )
#
# letter_cpd = TabularCPD(
#     variable='L',
#     variable_card=2,
#     values=[[0.1, 0.4, 0.99],
#             [0.9, 0.6, 0.01]],
#     evidence=['G'],
#     evidence_card=[3]
# )
#
# # 将包含概率表的各节点添加到模型中
# student_model.add_cpds(
#     grade_cpd,
#     difficulty_cpd,
#     intel_cpd,
#     letter_cpd,
#     sat_cpd
# )
# # 获得模型条件概率分布
# print(student_model.get_cpds())
# # 获得模型各个节点依赖关系
# print(student_model.active_trail_nodes('D'))
#
# # 进行贝叶斯推断
# from pgmpy.inference import VariableElimination
# student_infer = VariableElimination(student_model)
# prob_G = student_infer.query(variables=['G'])
# print(prob_G)
#
# prob_G = student_infer.query(
#             variables=['G'],
#             evidence={'I': 1, 'D': 0})
# print(prob_G)
#
# # 生成数据
# import numpy as np
# import pandas as pd
#
# raw_data = np.random.randint(low=0, high=2, size=(1000, 5))
# data = pd.DataFrame(raw_data, columns=['D', 'I', 'G', 'L', 'S'])
# data.head()
#
# #%%
#
# # 定义模型
# from pgmpy.models import BayesianModel
# from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
#
# model = BayesianModel([('D', 'G'), ('I', 'G'), ('I', 'S'), ('G', 'L')])
#
# # 基于极大似然估计进行模型训练
# model.fit(data, estimator=MaximumLikelihoodEstimator)
# for cpd in model.get_cpds():
#     # 打印条件概率分布
#     print("CPD of {variable}:".format(variable=cpd.variable))
#     print(cpd)
#
#
# #%%