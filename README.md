DecisionTree
============

决策树采用递归建树，建树过程中对分类任务使用增益率(gain ratio)作为度量指标，而回归任务使用MSE作为度量指标。

对于分类任务，选择能够使增益率最大的属性作为分割属性，特别的，对于连续属性需要枚举可以进行分割的点并二分计算增益率以得到最优的分割点。对于回归任务这一过程是类似的，不同之处在于总是选择使得MSE更小的属性作为分割属性。

缺失属性进行简单处理，在建树过程中将数据集中缺失属性进行替换：离散属性替换为该属性未丢失值的重数，连续属性替换为该属性未丢失值的平均数，并同时将使用的值记录以便于预测数据中出现缺失属性时进行替换。

另外在建树过程中进行了先剪枝工作，通过设定增益率阈值以及限制叶子结点样本数量的方法防止建树过程中的过度拟合。
