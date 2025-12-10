import numpy as np
class LogisticRegression:
    '''
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




    
    
    

