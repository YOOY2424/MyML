"""
MyML.Class 模块

本模块包含两个简单实现的机器学习模型：
- KNN: 基于 L_p 距离的 k 近邻分类器（二分类示例）
- LogisticRegression: 基于梯度下降的逻辑回归（二分类）

用途：
- 教学/练习用，演示基本的算法实现与 API（fit / predict / predict_proba）
- 若用于实际任务，请在性能、数值稳定性和边界条件上完善实现

使用示例：
>>> from Class import KNN, LogisticRegression
>>> knn = KNN(n_neighbors=5, p=2)
>>> knn.fit(X_train, y_train)
>>> y_pred = knn.predict(X_test)

>>> lr = LogisticRegression(learning_rate=0.1, num_k=1000)
>>> lr.fit(X_train, y_train)
>>> y_pred = lr.predict(X_test)

注意：
- 当前实现适用于 NumPy 数组输入，且假设 y 为二分类标签（0/1）。
- 对异常输入、单样本、类别不平衡、数值溢出等未做完整处理，可根据需要扩展。
"""
import numpy as np
###################
class KNN:
    '''
    n_neighbors是邻居数
    p是L_p距离的参数
    '''
    def __init__(
        self,
        n_neighbors=8,
        p=2
    ):
        self.n_neighbors=n_neighbors
        self.p=p
        self.x_train=None
        self.y_train=None
        self.sorted_data=None
    '''计算样本点之间的距离'''
    def L_p_distance(self,x_i,x_j,p=2):
        return np.sum((np.abs(x_i-x_j)**self.p))**(1/self.p)
    '''储存测试数据集'''
    def fit(self,X,y):
        self.x_train=np.array(X)
        self.y_train=np.array(y)
    '''对距离从小到大的排序'''
    def arr_sort(self,distances,y_labels):
        data = np.column_stack((distances,y_labels))
        n = len(data)
        for i in range(n-1):
            for j in range(n-i-1):
                if data[j,0] > data[j+1,1]:
                    data[[j,j+1]] = data[[j+1,j]]
        self.sorted_data=data
    '''进行标签的预测'''
    def predict(self,x_test):
        x_test = np.array(x_test)
        m,n = x_test.shape
        c,j = self.x_train.shape                
        y_proba_1 = []
        for k in range(m):
            #计算距离
            distances = np.sum((np.abs(self.x_train-x_test[k,:])**self.p))**(1/self.p)
            #对距离进行排序获得索引
            sort_d = np.argsort(distances)
            top_k_d = sort_d[:self.n_neighbors]
            top_K_label = self.y_train[top_k_d]
            #计算标签概率
            count = np.sum(top_K_label == 1)
            y_proba_1_ = count/self.n_neighbors
            y_proba_1.append(y_proba_1_)
        #打包概率
        y_proba_1 = np.array(y_proba_1)
        y_proba_0 = 1 - y_proba_1
        y_proba = np.column_stack((y_proba_0,y_proba_1))
        #根据概率打上标签
        y_pred = np.where(y_proba[:,1]>0.5,1,0)
        return  y_pred
###################
class LogisticRegression:
    '''
    w:线性函数的权重
    b:线性函数的偏置
    learning_rate:梯度下降的学习率 
    num_k:梯度下降迭代次数
    threshold:判断为正例的阈值，默认为0.5
    method:求取线性函数系数的方法
        默认变量:Gradient_Descent
    '''
    def __init__(
        self,
        learning_rate=0.01,
        num_k=1000,
        threshold = 0.5,
        method = 'Gradient_Descent'
    ):
        self.w=None
        self.b=0.0
        self.learning_rate=learning_rate
        self.num_k=num_k
        self.threshold = threshold
        self.method = method
    '''定义激活函数'''
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    '''梯度下降法求解系数'''
    def Gradient_Descent(self,X,y):
        m,n=X.shape
        self.w=np.zeros(n,)
        self.b=0.0
        for k in range(self.num_k): 
            z=np.dot(X,self.w)+self.b
            A = self.sigmoid(z) 
            dz =A-y 
            dw = 1/m*np.dot(X.T, dz)
            db = 1/m*np.sum(dz)
            self.w = self.w - self.learning_rate*dw
            self.b = self.b - self.learning_rate*db
    '''训练模型得到w和b，用的梯度下降法'''
    def fit(self,X,y):
        if self.method == 'Gradient_Descent':
            self.Gradient_Descent(X, y)
    '''预测数据'''
    def predict(self,x_test):
        return np.where(self.predict_proba(x_test)[:,1]>self.threshold,1,0)
    def predict_proba(self,x_test):
        z=np.dot(x_test,self.w)+self.b
        y_pred_1 = self.sigmoid(z)
        y_pred_0 = 1-y_pred_1
        y_pred_proba = np.column_stack((y_pred_0, y_pred_1))
        return y_pred_proba