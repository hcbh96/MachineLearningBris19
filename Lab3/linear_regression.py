"""Create a contour plot of a two-dimensional normal distribution

Parameters
__________
ax : acis handle to plot
mu : mean vector 2x1
Sigma : covariance matric 2x2

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_distribution(ax,mu,Sigma):
    x=np.linspace(-1.5, 1.5,100)
    x1p, x2p = np.meshgrid(x,x)
    pos = np.vstack((x1p.flatten(), x2p.flatten())).T
    pdf = np.random.multivariate_normal(mu.flatten(), Sigma)
    Z=pdf.pdf(pos)
    Z=Z.reshape(100,100)

    ax.contour(x1p,x2p,Z,5,colors='r',lw=5,alpha=0.7)
    ax.set_xlabel('w_0')
    ax.set_ylabel('x_1')

    return

#create prior distribution
tau=1*np.eye(2)
w_0=np.zeros((2,1))

#Create prior weights
#w = np.array([-1.3,0.3]).T

#Generate data X
#X_c=np.linspace(-1,1,10)
#X=[]
#for i in range(len(X_c)):
#    X.append([X_c[i], 1])

#Generate data Y
#computeY = lambda x: w.T*x+np.random.normal(0,0.3)
#Y = np.array([computeY(xi) for xi in X])

#Create fig
fig=plt.figure()
ax=fig.add_subplot(111)

#visualise distribution over w
plot_distribution(ax,w_0,tau)
fig.show()



