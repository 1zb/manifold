import numpy as np
import h5py
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
# import scipy.io

def load_coil(normalize=True):
    """ Load COIL20 data

    Args:

    Returns:
        data: N x D
        label: N x 1
    """
    f = h5py.File('coil20.mat','r')
    data = f.get('data')
    data = np.array(data).T
    labels = f.get('objlabel')
    labels = np.squeeze(np.array(labels))

    if normalize:
        data = (data - np.min(data)) / (np.max(data) - np.min(data))

    return (data, labels)

def calc_affinity(data, sigma=1, zeroify_diag=True, normalize=True):
    """ Calculate affinity matrixs

    Args:
        data: N x D

    Returns:
        euclid_dist: squared euclidean distance N x N
        gauss_aff: gaussian affinities N x N
    """
    euclid_dist = cdist(data, data) ** 2
    gauss_aff = np.exp(- euclid_dist / 2 / sigma)

    if zeroify_diag:
        np.fill_diagonal(euclid_dist, 0)
        np.fill_diagonal(gauss_aff, 0)

    if normalize:
        euclid_dist = euclid_dist / np.sum(euclid_dist)
        gauss_aff = gauss_aff / np.sum(gauss_aff)

    return (euclid_dist, gauss_aff)

def calc_ee_loss(output_data, attractive_weights, repulsive_weights, alpha):
    """ Calculate the ee loss function

    Args:
        output_data: low-dimensional data N x L
        attractive_weights: matrix N x N
        repulsive_weights: matrix N x N
        alpha: scalar lambda

    Returns:
        value of loss function
    """
    (euclid_dist, gauss_aff) = calc_affinity(output_data, sigma=0.5, zeroify_diag=False, normalize=False)
    value = np.sum(attractive_weights * euclid_dist + alpha * repulsive_weights * gauss_aff)

    return (value, euclid_dist, gauss_aff)

def get_laplacians(weights):
    """ Graph Laplacians

    Args:
        weights: N x N

    Returns:
        laplacian matrix: N x N
        degree matrix: N x N
    """
    degree = np.diag(np.sum(weights, axis=1))
    return (degree - weights, degree)

def ee_linear_search(output_data, attractive_weights, repulsive_weights, alpha, step_size, spectral_direction, gradients, loss):
    """ Backtracking Linear Search

    Args:

    Returns:

    """
    dummy = 0.1 * gradients.ravel().dot(spectral_direction.ravel())
    current_loss, _, gauss_aff = calc_ee_loss(output_data + step_size * spectral_direction, attractive_weights, repulsive_weights, alpha)
    while current_loss > loss + step_size * dummy:
        step_size  = 0.8 * step_size
        current_loss, _, gauss_aff = calc_ee_loss(output_data + step_size * spectral_direction, attractive_weights, repulsive_weights, alpha)

    output_data = output_data + step_size * spectral_direction
    return (output_data, current_loss, gauss_aff, step_size)
def ee(labels, attractive_weights, repulsive_weights, dim_low=2, alpha=1, num_iters=100):
    """ Elastic Embedding

    Args:

    Returns:

    """
    output_data = 1e-5 * np.random.randn(attractive_weights.shape[0], dim_low)
    loss_value, _, gauss_aff = calc_ee_loss(output_data, attractive_weights, repulsive_weights, alpha)
    print(loss_value)

    attractive_laplacians, attractive_degree = get_laplacians(attractive_weights)

    attractive_upper = np.linalg.cholesky(attractive_laplacians + 1e-10 * np.eye(attractive_laplacians.shape[0])).T
    step_size = 1
    for k in range(num_iters):
        repulsive_laplacians, _ = get_laplacians(repulsive_weights * gauss_aff)
        gradients = 4 * (attractive_laplacians - repulsive_laplacians).dot(output_data)
        spectral_direction = -np.linalg.solve(attractive_upper, np.linalg.solve(attractive_upper.T, gradients))

        output_data, loss_value, gauss_aff, step_size = ee_linear_search(output_data, attractive_weights, repulsive_weights, alpha, step_size, spectral_direction, gradients, loss_value)
        if np.linalg.norm(step_size * spectral_direction) < 1e-3:
            break
        print(loss_value, step_size)
    return (output_data)

def draw(data, labels):
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    marker = ['o', 'v', '^', '<', 's', 'p', '*', 'h', 'D', 'H']
    for i in range(11, 21):
        plt.plot(data[labels==i,0].flatten(), data[labels==i,1].flatten(), marker[i-11], c=c[i-11], markersize=20)
    # plt.scatter(data[:,0].flatten(), data[:,1].flatten(), c=labels, s=400)
    plt.legend(['class {0}'.format(i) for i in range(11, 21)], markerscale=0.7, loc='lower right')
    plt.show()
    return

if __name__ == '__main__':

    (data, labels) = load_coil()

    input_data = []
    input_label = []
    for i in range(11, 21):
        input_data.append(data[labels==i,:])
        input_label.append(labels[labels==i])
    input_data = np.vstack(input_data)
    input_label = np.hstack(input_label)


    repulsive_weights, attractive_weights = calc_affinity(input_data, sigma=30)

    output_data = ee(input_label, attractive_weights, repulsive_weights, alpha=10, num_iters=100)
    draw(output_data, input_label)
