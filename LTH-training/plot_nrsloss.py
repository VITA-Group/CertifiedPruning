from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
figure = plt.figure()
ax = Axes3D(figure)
#
# X = np.arange(-3,3,0.04)
# Y = np.arange(-1.5,1.5,0.04)
# X,Y = np.meshgrid(X,Y)
# Z = -np.tanh(1+X/(Y**2+1e-20))
# # Z = -np.tanh(1+X/(Y**0+1e-20))
# ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='rainbow')
# ax.set_xlabel('stability',fontsize=15)
# ax.yaxis.set_rotate_label('False')
# ax.set_ylabel('$\gamma$',fontsize=20,rotation=120)
# plt.savefig('./nrsloss.pdf',format='pdf')


X = np.arange(-3,3,0.04)
Y = np.arange(-3,3,0.04)
X,Y = np.meshgrid(X,Y)
Z = -np.tanh(1+X*Y/(4**2))
# Z = -np.tanh(1+X*Y)

ax.set_zlim(-1,1)
ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='gnuplot')
ax.set_xlabel('lower bound',fontsize=15)
ax.yaxis.set_rotate_label('False')
ax.set_ylabel('upper bound',fontsize=15,rotation=120)
plt.savefig('./nrsloss_gamma4.pdf',format='pdf')


plt.savefig('./asd.png')
