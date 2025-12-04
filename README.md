# MyML

这是一个用于教学和练习的轻量级机器学习示例仓库，包含手写的 KNN 和 LogisticRegression 简单实现，方便新手理解算法细节与基本 API。

## 内容简介
- Class.py: 包含 KNN 和 LogisticRegression 两个类的简单实现（fit / predict / predict_proba）。

## 使用示例
```python
import numpy as np
from Class import KNN, LogisticRegression

# 构造示例数据（异或式或任何二分类数据）
X_train = np.array([[0,0],[0,1],[1,0],[1,1]])
y_train = np.array([0,1,1,0])  # 示例标签

# KNN
knn = KNN(n_neighbors=3, p=2)
knn.fit(X_train, y_train)
X_test = np.array([[0,0],[1,1]])
print("KNN predict:", knn.predict(X_test))

# Logistic Regression
# 注意：LogisticRegression 假设数据是线性可分或可学习的
lr = LogisticRegression(learning_rate=0.1, num_k=1000)
lr.fit(X_train, y_train)
print("LR predict:", lr.predict(X_test))
```

## API（快速）
- KNN(n_neighbors=8, p=2)
  - fit(X, y)
  - predict(x_test)
  - predict_proba(x_test)
- LogisticRegression(learning_rate=0.01, num_k=1000, threshold=0.5)
  - fit(X, y)
  - predict(x_test)
  - predict_proba(x_test)

## 提交更改示例（本地）
```bash
git add Class.py README.md
git commit -m "Add module docstrings and README introduction"
git push origin main
```

## 进一步改进建议
- 对 KNN 和 LogisticRegression 代码做向量化与性能优化
- 增加输入校验、异常处理与单元测试
- 增加多分类支持、模型保存/加载功能
- KNN还有问题没有修改
