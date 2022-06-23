import tf_silent
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pinn import PINN
from network import Network
from optimizer import L_BFGS_B
import tensorflow as tf
import matplotlib.colors as colors

def u0(tx):
    """
    Initial wave form.
    Args:
        tx: variables (t, x) as tf.Tensor.
    Returns:
        u(t, x) as tf.Tensor.
    """

    t = tx[..., 0, None]
    x = tx[..., 1, None]


    return  0.25 * tf.sin(x)

if __name__ == '__main__':
    """
    Test the physics informed neural network (PINN) model for Burgers' equation
    """

    # number of training samples
    num_train_samples = 7000
    # number of test samples
    num_test_samples = 2000
    # kinematic viscosity
    nu = 0.001
    t_f = 4

    # build a core network model
    network = Network.build()
    network.summary()
    # build a PINN model
    pinn = PINN(network, nu).build()

    x_f = 2 * np.pi
    x_ini = 0

    # create training input
    tx_eqn = np.random.rand(num_train_samples, 2)          # t_eqn =  0 ~ +1
    tx_eqn[..., 1] = (x_f - x_ini) * tx_eqn[..., 1] + x_ini                # x_eqn = -1 ~ +1
    tx_eqn[..., 0] = t_f * tx_eqn[..., 0]
    tx_ini = (x_f - x_ini) * np.random.rand(num_train_samples, 2) + x_ini  # x_ini = -1 ~ +1
    tx_ini[..., 0] = 0                                     # t_ini =  0
    tx_bnd = np.random.rand(num_train_samples, 2)          # t_bnd =  0 ~ +1
    tx_bnd[..., 1] = (x_f - x_ini) * np.round(tx_bnd[..., 1]) + x_ini      # x_bnd = -1 or +1
    tx_bnd[..., 0] = t_f * tx_bnd[..., 0]
    # create training output
    u_eqn = np.zeros((num_train_samples, 1))               # u_eqn = 0
    u_ini = u0(tf.constant(tx_ini)).numpy()
    u_bnd = np.zeros((num_train_samples, 1))               # u_bnd = 0

    # train the model using L-BFGS-B algorithm
    x_train = [tx_eqn, tx_ini, tx_bnd]
    y_train = [u_eqn,  u_ini,  u_bnd]
    lbfgs = L_BFGS_B(model=pinn, x_train=x_train, y_train=y_train)
    lbfgs.fit()


    # predict u(t,x) distribution
    t_flat = np.linspace(0, t_f, num_test_samples)
    x_flat = np.linspace(x_ini, x_f, num_test_samples)
    t, x = np.meshgrid(t_flat, x_flat)
    tx = np.stack([t.flatten(), x.flatten()], axis=-1)
    u = network.predict(tx, batch_size=num_test_samples)
    u = u.reshape(t.shape)

    # plot u(t,x) distribution as a color-map
    fig = plt.figure(figsize=(15,10))
    plt.pcolormesh(t, x, u, cmap='rainbow')
    font1 = {'family':'serif','size':40}
    font2 = {'family':'serif','size':15}
    plt.xlabel('t', fontdict = font1)
    plt.ylabel('x', fontdict = font1)
    plt.title('u(t,x)', fontdict=font1)
    plt.tick_params(axis='both', which='major', labelsize=35)
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.mappable.set_clim(-1, 1)
    cbar.ax.tick_params(labelsize=35)
    plt.tight_layout()
    plt.show()

    # Solution at time 0, 2 and 4
    fig,(ax1, ax2, ax3)  = plt.subplots(1,3,figsize=(15,6))

    tx = np.stack([np.full(t_flat.shape, 0), x_flat], axis=-1)
    u_ = network.predict(tx, batch_size=num_test_samples)
    ax1.plot(x_flat, u_)
    font1 = {'family':'serif','size':20}
    font2 = {'family':'serif','size':15}
    ax1.set_title('t={}'.format(0), fontdict = font1)
    ax1.set_xlabel('x', fontdict = font1)
    ax1.set_ylabel('u(t,x)', fontdict = font1)
    ax1.tick_params(labelsize=15)

    tx = np.stack([np.full(t_flat.shape, 2), x_flat], axis=-1)
    u_ = network.predict(tx, batch_size=num_test_samples)
    ax2.plot(x_flat, u_)
    ax2.set_title('t={}'.format(2), fontdict = font1)
    ax2.set_xlabel('x', fontdict = font1)
    ax2.set_ylabel('u(t,x)', fontdict = font1)
    ax2.tick_params(labelsize=15)

    tx = np.stack([np.full(t_flat.shape, 4), x_flat], axis=-1)
    u_ = network.predict(tx, batch_size=num_test_samples)
    ax3.plot(x_flat, u_)
    ax3.set_title('t={}'.format(4), fontdict = font1)
    ax3.set_xlabel('x', fontdict = font1)
    ax3.set_ylabel('u(t,x)', fontdict = font1)
    ax3.tick_params(labelsize=15)
    plt.tight_layout()
    plt.show()

    #  Comparison with numerical solution
    import scipy.io

    data = scipy.io.loadmat('bathymetry_4.mat')
    Exact = np.matrix(data['data'])


    # predict u(t,x) distribution
    t_flat = np.linspace(0, t_f, 401)
    x_flat = np.linspace(x_ini, x_f, 400)
    t, x = np.meshgrid(t_flat, x_flat)
    tx = np.stack([t.flatten(), x.flatten()], axis=-1)
    u = network.predict(tx, batch_size=num_test_samples)
    u = u.reshape(t.shape)

    Error = (Exact.T - u)



    # plot u(t,x) distribution as a color-map
    fig = plt.figure(figsize=(15, 10))
    font1 = {'family': 'serif', 'size': 40}
    font2 = {'family': 'serif', 'size': 15}
    plt.pcolormesh(t, x, abs(Error), cmap='rainbow',norm = colors.LogNorm())
    plt.xlabel('t', fontdict=font1)
    plt.ylabel('x', fontdict=font1)
    plt.title('Error', fontdict=font1)
    plt.tick_params(axis='both', which='major', labelsize=35)
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.ax.tick_params(labelsize=35)

    plt.show()
