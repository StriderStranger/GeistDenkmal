# 实验展示4种算法调优Lasso超参数alpha：
# 1.LassoCV : 基于CoordinateDescent算法的k折交叉验证；
# 2.LassoLarsCV : 基于LeastAngleRegression算法的k折交叉验证；
# 3.LassoLarsIC : 基于AIC算法和BIC算法
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC
from sklearn import datasets

diabetes = datasets.load_diabetes()
X = diabetes.data                           # X.shape=(442,10)
y = diabetes.target
rng = np.random.RandomState(42)
X = np.c_[X, rng.randn(X.shape[0], 14)]     # add some bad features, X.shape=(442,24)
X /= np.sqrt(np.sum(X**2, axis=0))

# BIC Bayes Infomation criterion
model_bic = LassoLarsIC(criterion='bic')
t1 = time.time()
model_bic.fit(X,y)
t_bic = time.time() - t1
alpha_bic_ = model_bic.alpha_

# AIC Akaike Infomation criterion
model_aic = LassoLarsIC(criterion='aic')
model_aic.fit(X,y)
alpha_aic_ = model_aic.alpha_


def plot_ic_criterion(model, name, color):
    alpha_ = model.alpha_
    alphas_ = model.alphas_
    criterion_ = model.criterion_
    plt.plot(-np.log10(alphas_), criterion_, '--', color=color, 
            linewidth=3, label='%s criterion' % name)
    plt.axvline(-np.log10(alpha_), color=color, 
            linewidth=3, label='alpha: %s estimate' % name)
    plt.xlabel('-log(alpha)')
    plt.ylabel('criterion')

plt.figure()
plot_ic_criterion(model_aic, 'AIC', 'b')
plot_ic_criterion(model_bic, 'BIC', 'r')
plt.legend()
plt.title('Information-criterion for model selection (training time %.3fs)' % t_bic)

# CoordinateDescent算法
print("Computing regularization path using the coordinate descent lasso...")
t1 = time.time()
model = LassoCV(cv=20).fit(X,y)     # 20折验证
t_lasso_cv = time.time() - t1
m_log_alphas = -np.log10(model.alphas_)

plt.figure()
ymin,ymax = 2300, 3800
plt.plot(m_log_alphas, model.mse_path_, ':')
plt.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k',
        label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
        label='alpha: CV estimate')
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('Mean square error')
plt.title('Mean square error on each fold: coordinate descent '
          '(train time: %.2fs)' % t_lasso_cv)
plt.axis('tight')
plt.ylim(ymin,ymax)


# LeastAngleRegression算法
print("Computing regularization path using the Lars lasso...")
t1 = time.time()
model = LassoLarsCV(cv=20).fit(X, y)
t_lasso_lars_cv = time.time() - t1
m_log_alphas = -np.log10(model.cv_alphas_)

plt.figure()
plt.plot(m_log_alphas, model.cv_mse_path_, ':')
plt.plot(m_log_alphas, model.cv_mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('Mean square error')
plt.title('Mean square error on each fold: Lars (train time: %.2fs)'
          % t_lasso_lars_cv)
plt.axis('tight')
plt.ylim(ymin, ymax)

plt.show()