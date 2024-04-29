# This header file is modified from:
#   https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_tree.pxd


import numpy as np
cimport numpy as np

ctypedef np.npy_uint8 DTYPE_t               # Type of X (after binning)
ctypedef np.npy_float64 DOUBLE_t            # Type of y, sample_weight
ctypedef np.npy_int32 SIZE_t                # Type for counters, child, and feature ID
ctypedef np.npy_uint32 UINT32_t             # Unsigned 32 bit integer

from ._splitter cimport Splitter
from ._splitter cimport SplitRecord

cdef struct Node:
    # Base storage structure for the internal nodes in a Tree object (

    SIZE_t left_child                        # ID of the left child of the node
    SIZE_t right_child                       # ID of the right child of the node
    SIZE_t feature                           # Feature used for splitting the node
    DTYPE_t threshold                        # Threshold value at the node

cdef class Tree:
    # The Tree object is a binary tree structure constructed by the
    # TreeBuilder. The tree structure is used for predictions.

    # Input/Output layout
    cdef public SIZE_t n_features           # Number of features in X
    cdef SIZE_t* n_classes                  # Number of classes in y[:, k]
    cdef public SIZE_t n_outputs            # Number of outputs in y
    cdef public SIZE_t max_n_classes        # max(n_classes)

    # Inner structures: values are stored separately from node structure,
    # since size is determined at runtime.
    cdef public SIZE_t max_depth            # Max depth of the tree
    cdef public SIZE_t internal_node_count  # Counter for internal node IDs
    cdef public SIZE_t leaf_node_count      # Counter for leaf node IDS
    cdef public SIZE_t internal_capacity    # Capacity of internal nodes
    cdef public SIZE_t leaf_capacity        # Capacity of leaf nodes
    cdef Node* nodes                        # Array of internal nodes
    cdef double* value                      # Array of leaf nodes
    cdef SIZE_t value_stride                # = n_outputs * max_n_classes

    # Methods
    cdef SIZE_t _upd_parent(self, SIZE_t parent, bint is_left) except -1 nogil
    cdef SIZE_t _add_node(self, SIZE_t parent, bint is_left, bint is_leaf,
                          SIZE_t feature, DTYPE_t threshold) except -1 nogil
    cdef int _resize(self, SIZE_t internal_capacity,
                     SIZE_t leaf_capacity) except -1 nogil
    cdef int _resize_node_c(self, SIZE_t internal_capacity=*) except -1 nogil
    cdef int _resize_value_c(self, SIZE_t leaf_capacity=*) except -1 nogil

    cdef np.ndarray _get_value_ndarray(self)
    cdef np.ndarray _get_node_ndarray(self)

    cpdef np.ndarray predict(self, object X)

    cpdef np.ndarray apply(self, object X)
    cdef np.ndarray _apply_dense(self, object X)
    cdef np.ndarray _apply_sparse_csr(self, object X)

    cpdef object decision_path(self, object X)
    cdef object _decision_path_dense(self, object X)
    cdef object _decision_path_sparse_csr(self, object X)


# =============================================================================
# Tree builder
# =============================================================================

cdef class TreeBuilder:
    # The TreeBuilder recursively builds a Tree object from training samples,
    # using a Splitter object for splitting internal nodes and assigning
    # values to leaves.

    cdef Splitter splitter              # Splitting algorithm

    cdef SIZE_t min_samples_split       # Minimum number of samples in an internal node
    cdef SIZE_t min_samples_leaf        # Minimum number of samples in a leaf node
    cdef double min_weight_leaf         # Minimum weight in a leaf node
    cdef SIZE_t max_depth               # Maximal tree depth
    cdef double min_impurity_split
    cdef double min_impurity_decrease   # Impurity threshold for early stopping

    cpdef build(self, Tree tree, object X, np.ndarray y,
                np.ndarray sample_weight=*,
                np.ndarray X_idx_sorted=*)
    cdef _check_input(self, object X, np.ndarray y, np.ndarray sample_weight)
