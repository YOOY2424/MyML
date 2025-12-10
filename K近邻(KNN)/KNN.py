import numpy as np
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
