import tensorflow as tf
from nn import NN
import train
import numpy as np
from numpy.linalg import inv, eigh, norm
import os
from six.moves import cPickle as pkl
import time
from operator import sub
from matplotlib import pyplot as plt

def stats(X):
    L = []
    for x in X:
        mean = np.mean(np.abs(x))
        var = np.var(x)
        max = np.max(x)
        min = np.min(x)
        L.append([mean, var, max, min])
        print(L[-1])
    print('')
    return L

def test_correctness(sess, model, test_images, num_examples, num_samples):
    t = time.time()
    F1 = model.exact_fisher(sess, test_images[:num_examples], block_diag=False)
    print('elapsed time for exact_fisher: ' + str(time.time() - t))

    t = time.time()
    F2 = model.exact_fisher_with_samples(sess, test_images[:num_examples], num_samples, block_diag=False)
    print('elapsed time for exact_fisher_with_samples: ' + str(time.time() - t))

    stats(F1)
    stats(F2)
    stats(map(sub, F1, F2))

def compute_fisher(sess, model, test_images, num_examples):
    t = time.time()
    print('computing exact fisher')
    exact_fisher = model.exact_fisher(sess, test_images[:num_examples])
    print('elapsed time: ' + str(time.time() - t))

    t = time.time()
    print('computing kfac fisher')
    kfac_fisher = model.kfac_fisher(sess, test_images[:num_examples])
    print('elapsed time: ' + str(time.time() - t))

    t = time.time()
    print('diagonalizing exact_fisher')
    l_exact, P_exact = eigh(exact_fisher)
    print('elapsed time: ' + str(time.time() - t))

    t = time.time()
    print('changing basis for kfac_fisher')
    Fd = np.transpose(P_exact).dot(kfac_fisher).dot(P_exact)
    print('elapsed time: ' + str(time.time() - t))

    # t = time.time()
    # print('diagonalizing kfac_fisher')
    # l_kfac, P_kfac = eigh(kfac_fisher)
    # print('elapsed time: ' + str(time.time() - t))

    d = {'kfac_fisher': kfac_fisher,
         'exact_fisher': exact_fisher,
         'l_exact': l_exact,
         'P_exact': P_exact,
         'Fd': Fd }
         # 'l_kfac': l_kfac,
         # 'P_kfac': P_kfac}

    with open('result.pkl', 'wb') as f:
        pkl.dump(d, f)

    stats([exact_fisher])
    stats([kfac_fisher])

    return d

def compute_angles():
    with open('result.pkl', 'rb') as f:
        d = pkl.load(f)
    Fd = d['Fd']
    Fd[Fd < 1e-15] = 0

    l = d['l_exact']
    l[l < 1e-15] = 0
    L = np.diag(l)
    truncate = np.max(np.where(l==0)) + 1

    c, a, b = angles(Fd, L, truncate=truncate)
    with open('angles.pkl', 'wb') as f:
        pkl.dump(c, f)

    return c, a, b

def angles(A, B, truncate=0):
    '''return angles between columns of A and columns of B'''
    dot_prod = np.sum(A * B, axis = 0)
    A_norm = norm(A, axis = 0)
    B_norm = norm(B, axis = 0)

    norms = A_norm * B_norm
    norms = np.array([1. / x if x != 0 else 0 for x in list(norms)])

    angle = dot_prod * norms

    angle = angle[truncate:]
    angle = np.arccos(angle)
    return angle * 180 / np.pi

def compare():
    '''
    diagonalize inv(k)e and inv(e)k in symmetric form
    inv(A) and inv(B) are the respective change of basis matrices to put them in symmetric form
    '''
    with open('result.pkl', 'rb') as f:
        d = pkl.load(f)
    k = d['kfac_fisher']
    e = d['exact_fisher']

    perturb = np.zeros(e.shape[0], dtype=np.float32)
    perturb[:] = 1e-10
    e = e + np.diag(perturb)
    k = k + np.diag(perturb)

    l_k, P_k = np.linalg.eigh(k)
    l_e, P_e = np.linalg.eigh(e)

    A = P_k.dot(np.sqrt(np.diag(1. / l_k)))
    B = P_e.dot(np.sqrt(np.diag(1. / l_e)))

    N1 = (np.transpose(A)).dot(e).dot(A)
    N2 = (np.transpose(B)).dot(k).dot(B)

    #make symmetric
    N1 = (N1 + N1.T) / 2.
    N2 = (N2 + N2.T) / 2.

    l1, P1 = np.linalg.eigh(N1)
    l2, P2 = np.linalg.eigh(N2)

    ke = A.dot(P1)
    ek = B.dot(P2)
    ek = np.flip(ek, axis=1)

    #compute angles
    dot_prod = np.sum(ek * ke, axis=0)
    ek_norm = norm(ek, axis=0)
    ke_norm = norm(ke, axis=0)
    norms = ek_norm * ke_norm
    norms = np.array([1. / x if x != 0 else 0 for x in list(norms)])
    cos_angles = np.abs(dot_prod * norms)

    d = {'l1': l1,
         'P1': P1,
         'l2': l2,
         'P2': P2,
         'A': A,
         'B': B,
         'ek_norm': ek_norm,
         'ke_norm': ke_norm,
         'cos_angles': cos_angles}

    with open('compare.pkl', 'wb') as f:
        pkl.dump(d, f)

    # plt.plot(l1, 'ro', np.flip(1. /l2, axis=0), 'bo')
    # plt.savefig('eigenvalues.png')
    # plt.close()
    # plt.plot(ke_norm, 'ro', ek_norm, 'bo')
    # plt.savefig('norms.png')
    # plt.close()
    # plt.plot(dot_prod, 'ro')
    # plt.savefig('dot_product.png')
    # plt.close()
    # plt.plot(np.abs(cos_angles), 'ro')
    # plt.savefig('cos_angles.png')
    # plt.close()

    return d

def compare2():
    '''write kfac fisher in basis of N1 and then write
    the result in the basis of eigenvectors of N1
    then compare their columns
    '''
    with open('compare.pkl', 'rb') as f:
        d = pkl.load(f)
    with open('result.pkl', 'rb') as f:
        d1 = pkl.load(f)
    k = d1['kfac_fisher']
    k1 = np.linalg.inv(d['A']).dot(k).dot(d['A'])
    k2 = d['P1'].T.dot(k1).dot(d['P1'])
    L = np.diag(d['l1'])
    c, a, b = angles(k2, L, truncate = 0)
    plt.plot(c, 'ro')
    plt.savefig('angles_kf_N1.png')
    plt.close()
    return c, a, b

def kf_and_kfrev():
    data_dir = os.path.join(os.getcwd(), 'MNIST_data')
    with open(data_dir + '/test_images_16x16.pkl', 'rb') as f:
        test_images = pkl.load(f)
    sess, model = train.train_standard_model()
    kf = model.kfac_fisher(sess, test_images)
    kf_rev = model.kfac_fisher_rev(sess, test_images)

    plt.imshow(np.sqrt(np.abs(kf[5120:5520, 5120:5520])))
    plt.show()

    plt.imshow(np.sqrt(np.abs(kf_rev[5120:5520, 5120:5520])))
    plt.show()

    return kf, kf_rev

def covariances():
    data_dir = os.path.join(os.getcwd(), 'MNIST_data')
    with open(data_dir + '/test_images_16x16.pkl', 'rb') as f:
        test_images = pkl.load(f)
    sess, model = train.train_standard_model()

    cov_g, Eg1, Eg2 = model.g_covariance(sess, test_images)
    cov_a, Ea1, Ea2 = model.a_covariance(sess, test_images)

def compare3():
    '''
    change eigenvectors of N1 back to standard basis
    see how kf acts on them
    '''
    with open('compare.pkl', 'rb') as f:
        d = pkl.load(f)
    with open('result.pkl', 'rb') as f:
        d1 = pkl.load(f)
    kf = d1['kfac_fisher']
    A = d['A']
    P1 = d['P1']
    E1 = A.dot(P1)
    m = kf.dot(E1)
    c, a, b = angles(E1, kf.dot(E1), 0)
    plt.figure()
    plt.plot(c, 'o')
    plt.figure()
    plt.plot(a, 'o', b, 'o')
    plt.figure()
    plt.plot(m, 'o')
    plt.show()

def compare4():
    '''
    solve for eigenvectors of kf in terms of E1 (eigenvectors of N1 in std basis)
    '''
    with open('compare.pkl', 'rb') as f:
        d = pkl.load(f)
    with open('result.pkl', 'rb') as f:
        d1 = pkl.load(f)
    kf = d1['kfac_fisher']
    l, P = np.linalg.eigh(kf)
    A = d['A']
    P1 = d['P1']
    E1 = A.dot(P1)
    # perturb = np.zeros(kf.shape[0], dtype=np.float32)
    # perturb[:] = 1e-10
    # perturb = np.diag(perturb)
    X = np.linalg.solve(P,E1)
    plt.imshow(np.sqrt(np.abs(X)))
    plt.show()
    return X, P, E1

def make_k2():
    '''make modified kfac fisher by changing 'diagonal' entries to equal exact fisher'''
    #make mask
    h_dim = 20
    y_dim = 10
    x_dim = 256
    layers = [x_dim] + 3 * [h_dim] + [y_dim]
    sess = tf.Session()
    l = [(x1, x2) for x1, x2 in zip(layers, layers[1:])]
    A = []
    for n, i in enumerate(l):
        C = []
        for m, j in enumerate(l):
            if n==m:
                C.append(sess.run(NN(layers).kronecker_product(np.ones((i[0], i[0])), np.eye((i[1])))))
            else:
                C.append(np.zeros((i[0]*i[1], j[0]*j[1])))
        A.append(C)
    mask = np.asarray(np.bmat(A))

    #apply mask
    with open('result.pkl', 'rb') as f:
        d = pkl.load(f)
    f = d['exact_fisher']
    k = d['kfac_fisher']
    kh = np.multiply(mask, f) + np.multiply(1-mask, k)

    #diag fisher
    lf, Pf = eigh(f)

    #changing basis for k and kh
    k_d = np.transpose(Pf).dot(k).dot(Pf)
    kh_d = np.transpose(Pf).dot(kh).dot(Pf)

    L = np.diag(lf)
    a = angles(kh_d, L)
    b = angles(k_d, L)

def inverse():
    '''compare angles of columns of inverses in basis of eigenvectors of f'''
    with open('result.pkl', 'rb') as f:
        d = pkl.load(f)
    f = d['exact_fisher']
    k = d['kfac_fisher']
    l = d['l_exact']
    P = d['P_exact']
    perturb = np.zeros(f.shape[0], dtype=np.float32)
    perturb[:] = 1e-10
    f = f + np.diag(perturb)
    k = k + np.diag(perturb)

    #make mask
    h_dim = 20
    y_dim = 10
    x_dim = 256
    layers = [x_dim] + 3 * [h_dim] + [y_dim]
    sess = tf.Session()
    l = [(x1, x2) for x1, x2 in zip(layers, layers[1:])]
    A = []
    for n, i in enumerate(l):
        C = []
        for m, j in enumerate(l):
            if n==m:
                C.append(np.ones((i[0]*i[1], j[0]*j[1])))
            else:
                C.append(np.zeros((i[0]*i[1], j[0]*j[1])))
        A.append(C)
    mask = np.asarray(np.bmat(A))
    dk = np.multiply(mask, k)

    i_k = inv(k)
    i_dk = inv(dk)
    i_f = inv(f)
    i_P = np.flip(P, 1)

    ic_k = np.transpose(i_P).dot(i_k).dot(i_P)
    ic_dk = np.transpose(i_P).dot(i_dk).dot(i_P)

    ak = np.arccos(ic_k.diagonal()/norm(ic_k, axis=0)) * 180 / np.pi
    adk = np.arccos(ic_dk.diagonal()/norm(ic_dk, axis=0)) * 180 / np.pi

    plt.plot(ak, 'o', adk, 'o')
    plt.show()













