=====================================
Geometric Word Meta Embeddings (GWME)
=====================================

Generates Geometry-Aware word meta embeddings as discussed in:

Pratik Jawanpuria, N T V Satya Dev, Anoop Kunchukuttan, and Bamdev Mishra. `Learning geometric word
meta-embeddings <https://www.aclweb.org/anthology/2020.repl4nlp-1.6/>`_. In Proceedings of the 5th Workshop on Representation Learning for NLP, pages 39–44,
July 2020. Association for Computational Linguistics.



Abstract
========

We propose a geometric framework for learning meta-embeddings of words from different embedding sources. Our framework transforms the embeddings into a common latent
space, where, for example, simple averaging
or concatenation of different embeddings (of
a given word) is more amenable. The proposed latent space arises from two particular
geometric transformations - source embedding
specific orthogonal rotations and a common
Mahalanobis metric scaling. Empirical results
on several word similarity and word analogy
benchmarks illustrate the efficacy of the proposed framework

============
Requirements
============

The implementation of proposed Geometry framework is taken from `here <https://github.com/anoopkunchukuttan/geomm>`_ (GeoMM).
Checkout the GeoMM repo for installation of dependencies.

=====
Usage
=====

Assuming that the necessary PYTHON environment is already present.

- To generate latent space embeddings using the Geometry framework, run the following cmd

  - Checkout `help` for detailed info

.. code-block:: bash

   $ python geomm.py </path/to/emb1> </path/to/emb2> --dictionary "/path/to/dictionary" --normalize <normalize-criteria-for-emb1> <normalize-criteria-for-emb2> --max_opt_iter 150000 --l2_reg <value of regularizer> --geomm_embeddings_path </path/to/output-embeddings>

- To generate meta embeddings based on either `avg|conc` (Only methods supported now), run the following cmd

  - Checkout `help` for detailed info

.. code-block:: bash

   $ python generate-meta-embs.py --normalize <normalize-criteria-for-emb1> <normalize-criteria-for-emb2> --meta_embeddings_path </path/to/output-meta-embeddings>

**NOTE**

- The dictionary should be built as per the method specified in the paper which results in an identity matrix

Sample
======

A `sample <./sample/>`_ of 2 source embeddings and a bash `script <./generate-meta-embs.sh>`_ has been provided to generate meta-embeddings from them.

Run the following cmd (providing the exact cmds that are being run) for sample usecase

.. code-block:: bash

   $ bash -x generate-meta-embs.sh

