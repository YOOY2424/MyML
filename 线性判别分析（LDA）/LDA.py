import numpy as np
class LDA:
    def __init__(
        self,
        n_class=2,
        method = 'Byes'
    ):
        '''
        n_class:类别数
            默认是'2'
        method:计算阈值的方法
            默认是'Byes'
        '''
        self.n_class = n_class
        self.w = None
        self.method = method
        self.threshold = None
        self.u_0_score = None
        self.u_1_score = None
        self.p_0 = None
        self.p_1 = None
        self.sigma_sq = None
    def fit(self,X,y):
        """
        训练 LDA 模型
        X: (N_samples, n_features)
        y: (N_samples,)
        """
        if self.n_class == 2:
            # 1. 数据拆分
            # 使用布尔索引获取两个类别的数据
            x_0=X[y==0]
            x_1=X[y==1]
            # 2. 计算均值向量 (D,)
            u_0=np.mean(x_0, axis = 0)
            u_1=np.mean(x_1, axis = 0)
            # 3. 计算类内散度矩阵 Sw
            S_w = (x_0-u_0).T@(x_0-u_0) + (x_1-u_1).T@(x_1-u_1)
            # 4. 求解投影方向 w
            # w = Sw^(-1) * (u0 - u1)
            S_w += np.eye(S_w.shape[0]) * 1e-6
            self.w= np.linalg.solve(S_w, u_0-u_1)
            # 5. 计算投影均值、类别概率
            self.u_0_score = np.dot(self.w, u_0)
            self.u_1_score = np.dot(self.w, u_1)
            self.p_0 = len(y[y==0])/len(y)
            self.p_1 = len(y[y==1])/len(y)            
    def transform(self,x_train):
        """
        降维，返回投影后的数值
        """
        if self.w is None:
            raise Exception('请先Fit模型')            
        return np.dot(x_train,self.w )
    def predict(self,x_test):
        """
        返回类别标签 (0 或 1)
        """
        # 贝叶斯修正法
        # 降维后的x值关于阈值之间的判断        
        if self.method == 'Byes':
            # 计算投影后的方差、均值中值、贝叶斯修正项
            x_scores = self.transform(x_test)
            self.sigma_sq = np.sum(x_scores - np.mean(x_scores))
            mid_point = (self.u_0_score + self.u_1_score) / 2
            correction = (self.sigma_sq/(self.u_0_score - self.u_1_score))*np.log(self.p_0/self.p_1)
            # 计算阈值
            self.threshold = mid_point - correction            
            # w方向指向u_0，大于阈值为0类
            if self.u_0_score > self.u_1_score:
                y_pred = np.where(x_scores > self.threshold,0,1)
            # w方向指向u_1，大于阈值为1类
            else:
                y_pred = np.where(x_scores > self.threshold,1,0)     
        return y_pred
            
        
