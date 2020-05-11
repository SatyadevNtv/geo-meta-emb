# Code for GeoMM algorithm

import argparse
import collections
import numpy as np
import scipy.linalg
import sys
import time
import os
import theano.tensor as TT
from theano.sparse import as_sparse_or_tensor_variable, sub
from theano import shared
import datetime
from pymanopt import Problem
from pymanopt.manifolds import Stiefel, Product, PositiveDefinite, Euclidean
from pymanopt.solvers import ConjugateGradient
from scipy.sparse import coo_matrix
import gc

import embeddings

def normalize_emb(emb, method):
    """
    Normalize input embedding based on the choice of method
    """
    print(f"Normalizing using {method}")
    if method == 'unit':
        emb = embeddings.length_normalize(emb)
    elif method == 'center':
        emb = embeddings.mean_center(emb)
    elif method == 'unitdim':
        emb = embeddings.length_normalize_dimensionwise(emb)
    elif method == 'centeremb':
        emb = embeddings.mean_center_embeddingwise(emb)

    return emb


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate latent space embeddings')
    parser.add_argument('emb1', help='path to embedding 1')
    parser.add_argument('emb2', help='path to embedding 2')
    parser.add_argument('--geomm_embeddings_path', default=None, type=str, help='directory to save the output GeoMM latent space embeddings. The output embeddings are normalized.')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--verbose', default=0,type=int, help='Verbose')
    mapping_group = parser.add_argument_group('mapping arguments', 'Basic embedding mapping arguments')
    mapping_group.add_argument('--dictionary', default=sys.stdin.fileno(), help='the dictionary file (defaults to stdin)')
    mapping_group.add_argument('--normalize', choices=['unit', 'center', 'unitdim', 'centeremb', 'no'], nargs=2, default=[], help='the normalization actions performed in sequence for embeddings 1 and 2')
    
    geomm_group = parser.add_argument_group('GeoMM arguments', 'Arguments for GeoMM method')
    geomm_group.add_argument('--l2_reg', type=float,default=1e2, help='Lambda for L2 Regularization')
    geomm_group.add_argument('--max_opt_time', type=int,default=5000, help='Maximum time limit for optimization in seconds')
    geomm_group.add_argument('--max_opt_iter', type=int,default=150, help='Maximum number of iterations for optimization')
   
    args = parser.parse_args()
    
    if args.verbose:
        print('Current arguments: {0}'.format(args))

    dtype = 'float32'
    if args.verbose:
        print('Loading embeddings data...')

    # Read input embeddings
    emb1file = open(args.emb1, encoding=args.encoding, errors='surrogateescape')
    emb2file = open(args.emb2, encoding=args.encoding, errors='surrogateescape')
    emb1_words, x = embeddings.read(emb1file,max_voc=0, dtype=dtype)
    emb2_words, z = embeddings.read(emb2file,max_voc=0, dtype=dtype)

    # Build word to index map
    emb1_word2ind = {word: i for i, word in enumerate(emb1_words)}
    emb2_word2ind = {word: i for i, word in enumerate(emb2_words)}

    noov=0
    emb1_indices = []
    emb2_indices = []
    f = open(args.dictionary, encoding=args.encoding, errors='surrogateescape')
    for line in f:
        emb1,emb2 = line.split()
        try:
            emb1_ind = emb1_word2ind[emb1]
            emb2_ind = emb2_word2ind[emb2]
            emb1_indices.append(emb1_ind)
            emb2_indices.append(emb2_ind)
        except KeyError:
            noov+=1
            if args.verbose:
                print('WARNING: OOV dictionary entry ({0} - {1})'.format(emb1, emb2)) #, file=sys.stderr
    f.close()
    if args.verbose:
        print('Number of embedding pairs having at least one OOV: {}'.format(noov))
    emb1_indices = emb1_indices
    emb2_indices = emb2_indices
    if args.verbose:
        print('Normalizing embeddings...')

    # STEP 0: Normalization
    if len(args.normalize) > 0:
        x = normalize_emb(x, args.normalize[0])
        z = normalize_emb(z, args.normalize[1])

    # Step 1: Optimization
    if args.verbose:
        print('Beginning Optimization')
    start_time = time.time()
    x_count = len(set(emb1_indices))
    z_count = len(set(emb2_indices))

    # Filter out uniq values
    map_dict_emb1={}
    map_dict_emb2={}
    I=0
    uniq_emb1=[]
    uniq_emb2=[]
    for i in range(len(emb1_indices)):
        if emb1_indices[i] not in map_dict_emb1.keys():
            map_dict_emb1[emb1_indices[i]]=I
            I+=1
            uniq_emb1.append(emb1_indices[i])
    J=0
    for j in range(len(emb2_indices)):
        if emb2_indices[j] not in map_dict_emb2.keys():
            map_dict_emb2[emb2_indices[j]]=J
            J+=1
            uniq_emb2.append(emb2_indices[j])

    # Creating dictionary matrix
    row = list(range(0, x_count))
    col = list(range(0, x_count))
    data = [1 for i in range(0, x_count)]
    print(f"Counts: {x_count}, {z_count}")
    A = coo_matrix((data, (row, col)), shape=(x_count, z_count))

    np.random.seed(0)
    Lambda=args.l2_reg
    
    U1 = TT.matrix()
    U2 = TT.matrix()
    B  = TT.matrix()

    Xemb1 = x[uniq_emb1]
    Zemb2 = z[uniq_emb2]
    del x, z
    gc.collect()

    Kx, Kz = Xemb1, Zemb2
    XtAZ = Kx.T.dot(A.dot(Kz))
    XtX = Kx.T.dot(Kx)
    ZtZ = Kz.T.dot(Kz)
    AA = np.sum(A*A)

    W = (U1.dot(B)).dot(U2.T)
    regularizer = 0.5*Lambda*(TT.sum(B**2))
    sXtX = shared(XtX)
    sZtZ = shared(ZtZ)
    sXtAZ = shared(XtAZ)

    cost = regularizer
    wtxtxw = W.T.dot(sXtX.dot(W))
    wtxtxwztz = wtxtxw.dot(sZtZ)
    cost += TT.nlinalg.trace(wtxtxwztz)
    cost += -2 * TT.sum(W * sXtAZ)
    cost += shared(AA)

    solver = ConjugateGradient(maxtime=args.max_opt_time,maxiter=args.max_opt_iter)

    manifold =Product([Stiefel(Kx.shape[1], Kx.shape[1]), Stiefel(Kz.shape[1], Kz.shape[1]), PositiveDefinite(Kx.shape[1])])
    problem = Problem(manifold=manifold, cost=cost, arg=[U1, U2, B], verbosity=3)
    wopt = solver.solve(problem)
    print(f"Problem solved ...")

    w= wopt
    U1 = w[0]
    U2 = w[1]
    B = w[2]

    print(f"Model copied ...")

    gc.collect()

    # Step 2: Transformation
    xw = Kx.dot(U1).dot(scipy.linalg.sqrtm(B))
    zw = Kz.dot(U2).dot(scipy.linalg.sqrtm(B))
    print(f"Transformation done ...")

    end_time = time.time()
    if args.verbose:
        print('Completed training in {0:.2f} seconds'.format(end_time-start_time))

    del Kx, Kz, B, U1, U2
    gc.collect()

    ### Save the GeoMM embeddings if requested
    xw_n = embeddings.length_normalize(xw)
    zw_n = embeddings.length_normalize(zw)

    del xw, zw
    gc.collect()

    if args.geomm_embeddings_path is not None: 
        os.makedirs(args.geomm_embeddings_path,exist_ok=True)

        out_emb_fname=os.path.join(args.geomm_embeddings_path,'emb1.vec')
        new_emb1_words = []
        for id in uniq_emb1:
            new_emb1_words.append(emb1_words[id])
        with open(out_emb_fname,'w',encoding=args.encoding) as outfile:
            embeddings.write(new_emb1_words,xw_n,outfile)

        new_emb2_words = []
        for id in uniq_emb2:
            new_emb2_words.append(emb2_words[id])
        out_emb_fname=os.path.join(args.geomm_embeddings_path,'emb2.vec')
        with open(out_emb_fname,'w',encoding=args.encoding) as outfile:
            embeddings.write(new_emb2_words,zw_n,outfile)

    exit(0)

if __name__ == '__main__':
    main()
