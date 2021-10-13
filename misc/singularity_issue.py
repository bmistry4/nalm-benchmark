"""
Trying to recreate figure 2 from Neural arithmetic units paper (https://arxiv.org/pdf/2001.05016.pdf)
"""
import matplotlib.pyplot as plt
import numpy as np

##################################################
eps = 0.1
X = np.array([1, 1.2, 1.8, 2])
y = (X[0] + X[1]) * (X[0] + X[1] + X[2] + X[3])
interval_step = 0.01  # of mesh grid


###################################################

def rmse(y, y_hat):
    return np.power(np.power((y - y_hat), 2), 0.5)


def calc_num_steps(step_size):
    # assume range [-1,1]
    return int(2 * (1. / step_size) + 1)


"""
Using broadcasting
"""
# num_steps = calc_num_steps(interval_step)
# w1 = np.outer(np.linspace(-1, 1, num_steps), np.ones(num_steps))
# w2 = w1.copy().T  # transpose
#
# z1 = (X[0] * w1 + X[1] * w1)
# z2 = (X[0] * w1 + X[1] * w1 + X[2] * w1 + X[3] * w1)
# pred = np.power(np.abs(z1) + eps, w2) * np.power(np.abs(z2) + eps, w2)
# # rms_loss = ((y - y_hat) ** 2) ** 0.5
# rms_loss = rmse(y, pred)
#
# fig = plt.figure(figsize=(10, 6))
# ax = plt.axes(projection='3d')
# ax.plot_surface(w1, w2, rms_loss, cmap='magma', edgecolor='none')
# ax.set_title('Broadcasting')
# ax.set_xlabel('w1')
# ax.set_ylabel('w2')
# ax.set_zlabel('loss')
# # ax.view_init(45, -45)   # rotate plot


"""
Using np vectorized approach
"""


def calc_nac_mnac_loss(nac_w, mnac_w, eps):
    nac_W = np.array([[nac_w, nac_w, 0, 0], [nac_w, nac_w, nac_w, nac_w]])
    nac_out = np.matmul(X, nac_W.T)
    nac_out = np.abs(nac_out) + eps
    mnac_W = np.array([mnac_w, mnac_w])
    mnac_out = np.prod(nac_out ** mnac_W)
    return rmse(y, mnac_out)


def calc_nac_nmu_loss(nac_w, mnac_w, eps):
    nac_W = np.array([[nac_w, nac_w, 0, 0], [nac_w, nac_w, nac_w, nac_w]])
    nac_out = np.matmul(X, nac_W.T)
    nac_out = np.abs(nac_out) + eps
    nmu_W = np.array([mnac_w, mnac_w])
    nmu_out = np.prod(nac_out * nmu_W + 1 - nmu_W)
    return rmse(y, nmu_out)


def calc_nac_nru_loss(nac_w, mnac_w, eps):
    nac_W = np.array([[nac_w, nac_w, 0, 0], [nac_w, nac_w, nac_w, nac_w]])
    nac_out = np.matmul(X, nac_W.T)
    nac_out = np.abs(nac_out) + eps
    nru_W = np.array([mnac_w, mnac_w])
    nru_out = np.prod(np.sign(nac_out) * nac_out ** nru_W * np.abs(nru_W) + 1 - np.abs(nru_W))
    return rmse(y, nru_out)

def calc_nac_tanh_nru_loss(nac_w, mnac_w, eps):
    nac_W = np.array([[nac_w, nac_w, 0, 0], [nac_w, nac_w, nac_w, nac_w]])
    nac_out = np.matmul(X, nac_W.T)
    nac_out = np.abs(nac_out) + eps
    nru_W = np.array([mnac_w, mnac_w])
    nru_W_abs_approx = np.power(np.tanh(1000*nru_W),2)
    nru_out = np.prod(np.sign(nac_out) * np.power(nac_out, nru_W) * nru_W_abs_approx + 1 - nru_W_abs_approx)
    return rmse(y, nru_out)


w1 = np.arange(-1, 1, interval_step)
w2 = np.arange(-1, 1, interval_step)
W1, W2 = np.meshgrid(w1, w2)

loss_vectorised = np.vectorize(calc_nac_tanh_nru_loss)
loss = loss_vectorised(W1, W2, eps=eps)

fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(111, projection='3d')
mycmap = plt.get_cmap('Spectral')
ax1.set_title('Vectorised')
ax1.set_xlabel('w1')
ax1.set_ylabel('w2')
ax1.set_zlabel('loss')
surf1 = ax1.plot_surface(W1, W2, loss, cmap=mycmap)
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
ax1.view_init(45, -45)  # rotate plot
plt.show()
