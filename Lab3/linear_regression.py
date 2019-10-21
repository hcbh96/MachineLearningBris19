"""Create a contour plot of a two-dimensional normal distribution

Parameters
__________
ax : acis handle to plot
mu : mean vector 2x1
Sigma : covariance matric 2x2

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def plot_distribution(ax,mu,Sigma):
    x=np.linspace(-1.5, 1.5,100)
    x1p, x2p = np.meshgrid(x,x)
    pos = np.vstack((x1p.flatten(), x2p.flatten())).T
    pdf = multivariate_normal(mu.flatten(), Sigma)
    Z=pdf.pdf(pos)
    Z=Z.reshape(100,100)

    ax.contour(x1p,x2p,Z,5,colors='r',alpha=0.7)
    ax.set_xlabel('w_0')
    ax.set_ylabel('x_1')

    return

#create prior distribution
tau=1*np.eye(2)
w_0=np.zeros((2,1))

#Create fig
fig=plt.figure()
ax=fig.add_subplot(111)

#visualise distribution over w
plot_distribution(ax,w_0,tau)


#create fig 2
ax2=fig.add_subplot(222)

#Generate data X
X_r=np.linspace(-1,1,100)
convertToPoint = lambda x: [x, 1]
X=np.array([convertToPoint(xi) for xi in X_r])

#Create prior weights
w = np.array([-1.3,0.5]).T
beta=1/0.3

#Generate data Y
computeY = lambda x: w.T*x+np.random.normal(0,0.3)
Y = np.array([computeY(xi) for xi in X])

index = np.random.permutation(X.shape[0])
for i in range(0, index.shape[0]):
    #compute the posterior
    X_i = X[index,:]
    Y_i = Y[index]

    #compute posterior
    mu_1=(tau**(-1)+beta*X_i.T.dot(X_i))**(-1)*(tau.T.dot(w)+beta*X_i.T.dot(X_i))
    sigma_1=tau**(-1)+beta*X_i.T.dot(X_i)
    #visualise the posterior
    print(sigma_1)
    plot_distribution(ax2, mu_1, sigma_1)
    #visualise samples from posterior with data
    #print out the posterior mean

plt.show()

