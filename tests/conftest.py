"""
Set configs and other pre-test setup for tests in 
condo_rag/tests
"""

import os

# Limit native thread pools to avoid OpenMP / BLAS race conditions that can cause segfaults
# This problem is observed when running
# test_qa.py::Test_fixtures::test_indexA_1(), commit 83eedb3
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")