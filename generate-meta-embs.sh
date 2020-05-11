#!/bin/bash

set -euo pipefail

BINDIR="$(dirname $0)"

(
  mkdir -p "$BINDIR/sample/latent-embs"
  python geomm.py "$BINDIR/sample/src1.vec" "$BINDIR/sample/src2.vec" --dictionary "$BINDIR/sample/src1-src2.dict.txt" --normalize unitdim no --max_opt_iter 150000 --l2_reg 1e2 --geomm_embeddings_path "$BINDIR/sample/latent-embs/"
  python generate-meta-embs.py "$BINDIR/sample/latent-embs/emb1.vec" "$BINDIR/sample/latent-embs/emb2.vec" --normalize unitdim no --meta_embeddings_path "$BINDIR/sample/"
)
