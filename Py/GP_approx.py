# test for GPR slowfunction
import numpy as np
import matplotlib.pyplot as plt
import sys

# input array of numbers
# first input is # of train-point
# second is # of test-point

num = sys.argv

# number of training points
train_N = int(num[1])

# number of testing points
star_N = int(num[2])

# input lengthscale
ll = float(num[3])

# initial the test points randomly selected from [0,1]
#x_star = np.random.random(star_N)
#x_star.sort()
x_star = np.linspace(0, 1, star_N, endpoint = False)

def f(x):
	return np.exp((-1)/(x+0.00001))
	#return np.sin((0.5)*x)

def cov_fn(x1, x2, ll, s):
        return s**2 * np.exp((-1)*(x1-x2)**2/2/ll**2)

def covK(x_train, ll, s):
        N = x_train.shape[0]
	H = np.zeros((N,N))
        i = 0
        for k in x_train:
                H[i, :] = cov_fn(k, x_train, ll, s)
                i = i + 1
        return H

def covK_star(x_star, x_train, ll, s):
	H = np.zeros((x_star.shape[0], x_train.shape[0]))
	i = 0
	for k in x_star:
		H[i, :] = cov_fn(k, x_train, ll, s)
		i = i + 1
	return H

def kth_eigvec(k, N):
	if k == 0:
		return np.ones((N,1)) * (1/np.sqrt(N))
	else:
		a = np.zeros((N,1))
		a[0:k,0] = 1
		a[k,0] = (-1) *k
		return a * (1/np.sqrt(k**2 + k))

def predict_y(M, N, y_train, k_star, H):
	N = x_train.shape[0]
	M = x_star.shape[0]
	y_predict = np.zeros((M,1))
	for i in range(N):
		tmp = np.zeros((M,1))
		vec = kth_eigvec(i, N)
		lam_da = np.dot(np.matrix.transpose(vec),np.dot(H,vec))
		tmp = np.dot(np.matrix.transpose(vec),y_train) * np.dot(k_star, vec) * (1/lam_da)
		y_predict = y_predict + tmp
	return y_predict

def eig_app(N, H):
	tmp = np.zeros((N,1))
	for i in range(N):
		vec = kth_eigvec(i, N)
		tmp[i,0] = np.dot(np.matrix.transpose(vec),np.dot(H,vec))
	return tmp


x_train = np.linspace(0, 1, train_N, endpoint = False) 
y_train = f(x_train)
y_train = np.matrix.transpose(y_train)

s, s0 = 0.1, 0.01
#ll = 10

inv_inv_K = s0**2 * np.eye(train_N) + covK(x_train, ll, s) 

#b = np.linalg.eig(inv_inv_K)
#c = eig_app(x_train.shape[0], inv_inv_K)
#plt.plot(b[0],'r-')
#plt.plot(c,'bo')
#plt.show()
 
#inv_K = np.linalg.inv(inv_inv_K)

k_star = covK_star(x_star, x_train, ll, s)

y_star = predict_y(x_star.shape[0], x_train.shape[0], y_train, k_star, inv_inv_K)

#y__star = np.dot(inv_K, y_train)
#y_star = np.dot(k_star, y__star)

plt.plot(x_train, y_train)
plt.plot(x_star, y_star)
plt.show()


