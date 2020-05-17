========
Overview
========

Generates Geometry-Aware word meta embeddings as per `this <https://arxiv.org/abs/2004.09219>`_ work

- The base implementation for Geometry related framework is picked from `here <https://github.com/anoopkunchukuttan/geomm>`_ (GeoMM) which has a similar setting as cited in above paper.

============
Requirements
============

Install the `requirements.txt` into your python env

.. code-block:: bash

   $ python3 -m venv env
   $ source env/bin/activate
   $ pip install -r requirements.txt

=====
Usage
=====

Assuming that the necessary PYTHON environment is already present (by `source env/bin/activate`).

- The zip folder contains `./libs`. This needs to be used to include `pymanopt`

  .. code-block:: bash

     $ PYTHONPATH="$PYTHONPATH":./libs/ # Either export this environemnt/prepend this with the `python` cmds which are being executed.

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
- The `requirements.txt` is the minimal version needed to execute the cmds. For example, if `pymanopt` is installed via `pip`, it downloads `torch`
  to do efficient computations.

Sample
======

A `sample <./sample/>`_ of 2 source embeddings and a bash `script <./generate-meta-embs.sh>`_ has been provided to generate meta-embeddings from them.

Run the following cmd (providing the exact cmds that are being run) for sample usecase

.. code-block:: bash

   $ bash -x generate-meta-embs.sh

