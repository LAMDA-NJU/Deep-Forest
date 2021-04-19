# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# This class is modified from:
#   https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_tree.pyx


from cpython cimport Py_INCREF, PyObject, PyTypeObject

from libc.stdlib cimport free
from libc.string cimport memcpy


from libc.string cimport memset
from libc.stdint cimport SIZE_MAX

import numpy as np
cimport numpy as np
np.import_array()

from scipy.sparse import issparse
from scipy.sparse import csr_matrix

from ._utils cimport Stack
from ._utils cimport StackRecord
from ._utils cimport safe_realloc
from ._utils cimport sizet_ptr_to_ndarray

cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(PyTypeObject* subtype, np.dtype descr,
                                int nd, np.npy_intp* dims,
                                np.npy_intp* strides,
                                void* data, int flags, object obj)

# =============================================================================
# Types and constants
# =============================================================================

from numpy import uint8 as DTYPE
from numpy import float64 as DOUBLE

cdef double INFINITY = np.inf
cdef double EPSILON = np.finfo('double').eps

TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef SIZE_t _TREE_LEAF = TREE_LEAF
cdef SIZE_t _TREE_UNDEFINED = TREE_UNDEFINED
cdef SIZE_t INITIAL_STACK_SIZE = 10

# Repeat struct definition for numpy
NODE_DTYPE = np.dtype({
    'names': ['left_child', 'right_child', 'feature', 'threshold'],
    'formats': [np.int32, np.int32, np.int32, np.uint8],
    'offsets': [
        <Py_ssize_t> &(<Node*> NULL).left_child,
        <Py_ssize_t> &(<Node*> NULL).right_child,
        <Py_ssize_t> &(<Node*> NULL).feature,
        <Py_ssize_t> &(<Node*> NULL).threshold
    ]
})

# =============================================================================
# TreeBuilder
# =============================================================================

cdef class TreeBuilder:
    """Interface for different tree building strategies."""

    cpdef build(self, Tree tree, object X, np.ndarray y,
                np.ndarray sample_weight=None,
                np.ndarray X_idx_sorted=None):
        """Build a decision tree from the training set (X, y)."""
        pass

    cdef inline _check_input(self, object X, np.ndarray y,
                             np.ndarray sample_weight):
        """Check input dtype, layout and format"""
        if issparse(X):
            X = X.tocsc()
            X.sort_indices()

            if X.data.dtype != DTYPE:
                X.data = np.ascontiguousarray(X.data, dtype=DTYPE)

            if X.indices.dtype != np.int32 or X.indptr.dtype != np.int32:
                raise ValueError("No support for np.int64 index based "
                                 "sparse matrices")

        elif X.dtype != DTYPE:
            # since we have to copy we will make it fortran for efficiency
            X = np.asfortranarray(X, dtype=DTYPE)

        if y.dtype != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if (sample_weight is not None and
            (sample_weight.dtype != DOUBLE or
            not sample_weight.flags.contiguous)):
                sample_weight = np.asarray(sample_weight, dtype=DOUBLE,
                                           order="C")

        return X, y, sample_weight

# Depth first builder ---------------------------------------------------------

cdef class DepthFirstTreeBuilder(TreeBuilder):
    """Build a decision tree in depth-first fashion."""

    def __cinit__(self, Splitter splitter, SIZE_t min_samples_split,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  SIZE_t max_depth, double min_impurity_decrease,
                  double min_impurity_split):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split

    cpdef build(self, Tree tree, object X, np.ndarray y,
                np.ndarray sample_weight=None,
                np.ndarray X_idx_sorted=None):
        """Build a decision tree from the training set (X, y)."""

        # check input
        X, y, sample_weight = self._check_input(X, y, sample_weight)

        cdef DOUBLE_t* sample_weight_ptr = NULL
        if sample_weight is not None:
            sample_weight_ptr = <DOUBLE_t*> sample_weight.data

        # Initial capacity
        cdef int init_internal_capacity
        cdef int init_leaf_capacity

        if tree.max_depth <= 10:
            init_internal_capacity = (2 ** (tree.max_depth + 1)) - 1
            init_leaf_capacity = (2 ** (tree.max_depth + 1)) - 1
        else:
            init_internal_capacity = 2047
            init_leaf_capacity = 2047

        tree._resize(init_internal_capacity, init_leaf_capacity)

        # Parameters
        cdef Splitter splitter = self.splitter
        cdef SIZE_t max_depth = self.max_depth
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef SIZE_t min_samples_split = self.min_samples_split
        cdef double min_impurity_decrease = self.min_impurity_decrease
        cdef double min_impurity_split = self.min_impurity_split

        # Recursive partition (without actual recursion)
        splitter.init(X, y, sample_weight_ptr, X_idx_sorted)

        cdef SIZE_t start
        cdef SIZE_t end
        cdef SIZE_t depth
        cdef SIZE_t parent
        cdef bint is_left
        cdef SIZE_t n_node_samples = splitter.n_samples
        cdef double weighted_n_samples = splitter.weighted_n_samples
        cdef double weighted_n_node_samples
        cdef SplitRecord split
        cdef SIZE_t node_id

        cdef double impurity = INFINITY
        cdef SIZE_t n_constant_features
        cdef bint is_leaf
        cdef bint first = 1
        cdef SIZE_t max_depth_seen = -1
        cdef int rc = 0

        cdef Stack stack = Stack(INITIAL_STACK_SIZE)
        cdef StackRecord stack_record

        with nogil:
            # push root node onto stack
            # {start, end, depth, parent, is_left, impurity, n_constant_features}
            rc = stack.push(0, n_node_samples, 0, _TREE_UNDEFINED, 0, INFINITY, 0)
            if rc == -1:
                # got return code -1: out-of-memory
                with gil:
                    raise MemoryError()

            while not stack.is_empty():
                stack.pop(&stack_record)

                start = stack_record.start
                end = stack_record.end
                depth = stack_record.depth
                parent = stack_record.parent
                is_left = stack_record.is_left
                impurity = stack_record.impurity
                n_constant_features = stack_record.n_constant_features

                n_node_samples = end - start
                splitter.node_reset(start, end, &weighted_n_node_samples)

                is_leaf = (depth >= max_depth or
                           n_node_samples < min_samples_split or
                           n_node_samples < 2 * min_samples_leaf or
                           weighted_n_node_samples < 2 * min_weight_leaf)

                if first:
                    impurity = splitter.node_impurity()
                    first = 0

                is_leaf = (is_leaf or
                           (impurity <= min_impurity_split))

                if not is_leaf:
                    splitter.node_split(impurity, &split, &n_constant_features)
                    # If EPSILON=0 in the below comparison, float precision
                    # issues stop splitting, producing trees that are
                    # dissimilar to v0.18
                    is_leaf = (is_leaf or split.pos >= end or
                               (split.improvement + EPSILON <
                                min_impurity_decrease))

                if not is_leaf:
                    # Add internal nodes
                    node_id = tree._add_node(parent, is_left, is_leaf,
                                             split.feature, split.threshold)

                    # Push right child on stack
                    rc = stack.push(split.pos, end, depth + 1, node_id, 0,
                                    split.impurity_right, n_constant_features)
                    if rc == -1:
                        break

                    # Push left child on stack
                    rc = stack.push(start, split.pos, depth + 1, node_id, 1,
                                    split.impurity_left, n_constant_features)
                    if rc == -1:
                        break
                else:
                    # Update the parent nodes of leaf nodes
                    node_id = tree._upd_parent(parent, is_left)

                    # Set values for leaf nodes
                    splitter.node_value(tree.value + node_id * tree.value_stride)

                if node_id == SIZE_MAX:
                    rc = -1
                    break

                if depth > max_depth_seen:
                    max_depth_seen = depth

            if rc >= 0 and tree.internal_node_count > 0:
                rc = tree._resize_node_c(tree.internal_node_count)

            if rc >= 0:
                rc = tree._resize_value_c(tree.leaf_node_count)

            if rc >= 0:
                tree.max_depth = max_depth_seen

        if rc == -1:
            raise MemoryError()

# =============================================================================
# Tree
# =============================================================================

cdef class Tree:
    """Array-based representation of a binary decision tree.

    The binary tree is represented as a number of parallel arrays. The i-th
    element of each array holds information about the node `i`. Node 0 is the
    tree's root. You can find a detailed description of all arrays in
    `_tree.pxd`. NOTE: Some of the arrays only apply to either leaves or split
    nodes, resp. In this case the values of nodes of the other type are
    arbitrary!

    Attributes
    ----------
    internal_node_count : int
        The number of internal nodes in the tree.

    internal_capacity : int
        The current capacity (i.e., size) of the array that stores internal
        nodes, which is at least as great as `internal_node_count`.

    leaf_node_count : int
        The number of leaf nodes in the tree.

    leaf_capacity : int
        The current capacity (i.e., size) of the array that stores leaf
        nodes, which is at least as great as `leaf_capacity`.

    max_depth : int
        The depth of the tree, i.e. the maximum depth of its leaves.

    children_left : array of int, shape [internal_node_count]
        children_left[i] holds the node id of the left child of node i.
        For leaves, children_left[i] == TREE_LEAF. Otherwise,
        children_left[i] > i. This child handles the case where
        X[:, feature[i]] <= threshold[i].

    children_right : array of int, shape [internal_node_count]
        children_right[i] holds the node id of the right child of node i.
        For leaves, children_right[i] == TREE_LEAF. Otherwise,
        children_right[i] > i. This child handles the case where
        X[:, feature[i]] > threshold[i].

    feature : array of int, shape [internal_node_count]
        feature[i] holds the feature to split on, for the internal node i.

    threshold : array of double, shape [internal_node_count]
        threshold[i] holds the threshold for the internal node i.

    value : array of double, shape [leaf_node_count, n_outputs, max_n_classes]
        Contains the constant prediction value of each leaf node.
    """
    # Wrap for outside world.
    # WARNING: these reference the current `nodes` and `value` buffers, which
    # must not be freed by a subsequent memory allocation.
    # (i.e. through `_resize` or `__setstate__`)
    property n_classes:
        def __get__(self):
            return sizet_ptr_to_ndarray(self.n_classes, self.n_outputs)

    property children_left:
        def __get__(self):
            return self._get_node_ndarray()['left_child'][:self.internal_node_count]

    property children_right:
        def __get__(self):
            return self._get_node_ndarray()['right_child'][:self.internal_node_count]

    property n_internals:
        def __get__(self):
            return self.internal_node_count

    property n_leaves:
        def __get__(self):
            return self.leaf_node_count

    property feature:
        def __get__(self):
            return self._get_node_ndarray()['feature'][:self.internal_node_count]

    property threshold:
        def __get__(self):
            return self._get_node_ndarray()['threshold'][:self.internal_node_count]

    property value:
        def __get__(self):
            return self._get_value_ndarray()[:self.leaf_node_count]

    def __cinit__(self, int n_features, np.ndarray[SIZE_t, ndim=1] n_classes,
                  int n_outputs):
        """Constructor."""
        # Input/Output layout
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_classes = NULL
        safe_realloc(&self.n_classes, n_outputs)

        self.max_n_classes = np.max(n_classes)
        self.value_stride = n_outputs * self.max_n_classes

        cdef SIZE_t k
        for k in range(n_outputs):
            self.n_classes[k] = n_classes[k]

        # Inner structures
        self.max_depth = 0
        self.internal_node_count = 0
        self.leaf_node_count = 0
        self.internal_capacity = 0
        self.leaf_capacity = 0
        self.value = NULL
        self.nodes = NULL

    def __dealloc__(self):
        """Destructor."""
        # Free all inner structures
        free(self.n_classes)
        free(self.value)
        free(self.nodes)

    def __reduce__(self):
        """Reduce re-implementation, for pickling."""
        return (Tree, (self.n_features,
                       sizet_ptr_to_ndarray(self.n_classes, self.n_outputs),
                       self.n_outputs), self.__getstate__())

    def __getstate__(self):
        """Getstate re-implementation, for pickling."""
        d = {}
        d["max_depth"] = self.max_depth
        d["internal_node_count"] = self.internal_node_count
        d["leaf_node_count"] = self.leaf_node_count
        d["nodes"] = self._get_node_ndarray()
        d["values"] = self._get_value_ndarray()
        return d

    def __setstate__(self, d):
        """Setstate re-implementation, for unpickling."""
        self.max_depth = d["max_depth"]
        self.internal_node_count = d["internal_node_count"]
        self.leaf_node_count = d["leaf_node_count"]

        if 'nodes' not in d:
            raise ValueError('You have loaded Tree version which '
                             'cannot be imported')

        node_ndarray = d['nodes']
        value_ndarray = d['values']

        value_shape = (value_ndarray.shape[0], self.n_outputs,
                       self.max_n_classes)

        if (node_ndarray.ndim != 1 or
                node_ndarray.dtype != NODE_DTYPE or
                not node_ndarray.flags.c_contiguous or
                value_ndarray.shape != value_shape or
                not value_ndarray.flags.c_contiguous or
                value_ndarray.dtype != np.float64):
            raise ValueError('Did not recognise loaded array layout')

        if self._resize_value_c(self.leaf_node_count) != 0:
            raise MemoryError("Failure on resizing leaf nodes to %d" %
                              self.leaf_node_count)

        value = memcpy(self.value, (<np.ndarray> value_ndarray).data,
                       self.leaf_node_count * self.value_stride * sizeof(double))

        if self._resize_node_c(self.internal_node_count) != 0:
            raise MemoryError("Failure on resizing internal nodes to %d" %
                              self.internal_node_count)

        nodes = memcpy(self.nodes, (<np.ndarray> node_ndarray).data,
                       self.internal_node_count * sizeof(Node))

    cdef int _resize(self, SIZE_t internal_capacity,
                     SIZE_t leaf_capacity) nogil except -1:
        """Resize `self.nodes` to `internal_capacity`, and resize `self.value`
        to `leaf_capacity`.

        Returns -1 in case of failure to allocate memory (and raise
        MemoryError) or 0 otherwise.
        """
        if self._resize_node_c(internal_capacity) != 0:
            # Acquire gil only if we need to raise
            with gil:
                raise MemoryError("Failure on resizing internal nodes to %d" %
                                  internal_capacity)

        if self._resize_value_c(leaf_capacity) != 0:
            # Acquire gil only if we need to raise
            with gil:
                raise MemoryError("Failure on resizing leaf nodes to %d" %
                                  leaf_capacity)

    cdef int _resize_node_c(self,
                            SIZE_t internal_capacity=SIZE_MAX) nogil except -1:
        """Resize `self.nodes` to `internal_capacity`.

        Returns -1 in case of failure to allocate memory (and raise
        MemoryError) or 0 otherwise.
        """

        if internal_capacity == self.internal_capacity and self.nodes != NULL:
            return 0

        if internal_capacity == SIZE_MAX:
            if self.internal_capacity == 0:
                internal_capacity = 3  # default initial value
            else:
                internal_capacity = 2 * self.internal_capacity

        safe_realloc(&self.nodes, internal_capacity)

        if internal_capacity < self.internal_node_count:
            self.internal_node_count = internal_capacity

        self.internal_capacity = internal_capacity
        return 0

    cdef int _resize_value_c(self,
                             SIZE_t leaf_capacity=SIZE_MAX) nogil except -1:
        """Resize `self.value` to `leaf_capacity`.
        
        Returns -1 in case of failure to allocate memory (and raise
        MemoryError) or 0 otherwise.
        """

        if leaf_capacity == self.leaf_capacity and self.value != NULL:
            return 0

        if leaf_capacity == SIZE_MAX:
            if self.leaf_capacity == 0:
                leaf_capacity = 3  # default initial value
            else:
                leaf_capacity = 2 * self.leaf_capacity

        safe_realloc(&self.value, leaf_capacity * self.value_stride)

        # value memory is initialised to 0 to enable classifier argmax
        if leaf_capacity > self.leaf_capacity:
            memset(<void*>(self.value + self.leaf_capacity * self.value_stride),
                   0, (leaf_capacity - self.leaf_capacity) *
                   self.value_stride * sizeof(double))

        if leaf_capacity < self.leaf_node_count:
            self.leaf_node_count = leaf_capacity

        self.leaf_capacity = leaf_capacity
        return 0

    cdef SIZE_t _upd_parent(self, SIZE_t parent, bint is_left) nogil except -1:
        """Add a leaf node to the tree and connect it with its parent. Notice
        that `self.nodes` does not store any information on leaf nodes except
        the id of leaf nodes. In addition, the id of leaf nodes are multiplied
        by `_TREE_LEAF` to distinguish them from the id of internal nodes.
        
        The generated node id will be used to set `self.value` later.
        
        Returns (size_t)(-1) on error.
        """
        cdef SIZE_t node_id = self.leaf_node_count

        if node_id >= self.leaf_capacity:
            if self._resize_value_c() != 0:
                return SIZE_MAX

        if parent != _TREE_UNDEFINED:
            if is_left:
                self.nodes[parent].left_child = _TREE_LEAF * node_id
            else:
                self.nodes[parent].right_child = _TREE_LEAF * node_id

        self.leaf_node_count += 1

        return node_id

    cdef SIZE_t _add_node(self, SIZE_t parent, bint is_left, bint is_leaf,
                          SIZE_t feature, DTYPE_t threshold) nogil except -1:
        """Add an internal node to the tree.

        The new node registers itself as the child of its parent.

        Returns (size_t)(-1) on error.
        """
        cdef SIZE_t node_id = self.internal_node_count

        if node_id >= self.internal_capacity:
            if self._resize_node_c() != 0:
                return SIZE_MAX

        cdef Node* node = &self.nodes[node_id]

        if parent != _TREE_UNDEFINED:
            if is_left:
                self.nodes[parent].left_child = node_id
            else:
                self.nodes[parent].right_child = node_id

        # left_child and right_child will be set later
        node.feature = feature
        node.threshold = threshold

        self.internal_node_count += 1

        return node_id

    cpdef np.ndarray predict(self, object X):
        """Predict target for X."""
        out = self._get_value_ndarray().take(self.apply(X), axis=0, mode='clip')
        if self.n_outputs == 1:
            out = out.reshape(X.shape[0], self.max_n_classes)
        return out

    cpdef np.ndarray apply(self, object X):
        """Finds the terminal region (=leaf node) for each sample in X."""

        if issparse(X):
            return self._apply_sparse_csr(X)
        else:
            return self._apply_dense(X)

    cdef np.ndarray _apply_dense(self, object X):
        """Finds the terminal region (=leaf node) for each sample in X."""

        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.uint8, got %s" % X.dtype)

        # Extract input
        cdef const DTYPE_t[:, :] X_ndarray = X
        cdef SIZE_t n_samples = X.shape[0]

        # Initialize output
        cdef np.ndarray[SIZE_t] out = np.zeros((n_samples,), dtype=np.int32)
        cdef SIZE_t* out_ptr = <SIZE_t*> out.data

        # Initialize auxiliary data-structure
        cdef Node* node = NULL
        cdef SIZE_t i = 0
        cdef SIZE_t node_id = 0

        with nogil:
            for i in range(n_samples):
                node = self.nodes
                node_id = 0

                # While one of the two children of the current node is not a
                # leaf node
                while node.left_child > 0 or node.right_child > 0:

                    # If the left child is a leaf node
                    if node.left_child <= 0:

                        # If X[i] should be assigned to the left child
                        if X_ndarray[i, node.feature] <= node.threshold:
                            out_ptr[i] = <SIZE_t>(_TREE_LEAF * node.left_child)
                            break
                        else:
                            node_id = node.right_child
                            node = &self.nodes[node.right_child]
                            continue

                    # If the right child is a leaf node
                    if node.right_child <= 0:

                        # If X[i] should be assigned to the right child
                        if X_ndarray[i, node.feature] > node.threshold:
                            out_ptr[i] = <SIZE_t>(_TREE_LEAF * node.right_child)
                            break
                        else:
                            node_id = node.left_child
                            node = &self.nodes[node.left_child]
                            continue

                    # If the left and right child are both internal nodes
                    if X_ndarray[i, node.feature] <= node.threshold:
                        node_id = node.left_child
                        node = &self.nodes[node.left_child]
                    else:
                        node_id = node.right_child
                        node = &self.nodes[node.right_child]

                # If the left and child child are both leaf nodes
                if node.left_child <= 0 and node.right_child <= 0:
                    if X_ndarray[i, node.feature] <= node.threshold:
                        out_ptr[i] = <SIZE_t>(_TREE_LEAF * node.left_child)
                    else:
                        out_ptr[i] = <SIZE_t>(_TREE_LEAF * node.right_child)

        return out

    cdef inline np.ndarray _apply_sparse_csr(self, object X):
        """Finds the terminal region (=leaf node) for each sample in sparse X.
        """
        # Check input
        if not isinstance(X, csr_matrix):
            raise ValueError("X should be in csr_matrix format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef np.ndarray[ndim=1, dtype=DTYPE_t] X_data_ndarray = X.data
        cdef np.ndarray[ndim=1, dtype=SIZE_t] X_indices_ndarray  = X.indices
        cdef np.ndarray[ndim=1, dtype=SIZE_t] X_indptr_ndarray  = X.indptr

        cdef DTYPE_t* X_data = <DTYPE_t*>X_data_ndarray.data
        cdef SIZE_t* X_indices = <SIZE_t*>X_indices_ndarray.data
        cdef SIZE_t* X_indptr = <SIZE_t*>X_indptr_ndarray.data

        cdef SIZE_t n_samples = X.shape[0]
        cdef SIZE_t n_features = X.shape[1]

        # Initialize output
        cdef np.ndarray[SIZE_t, ndim=1] out = np.zeros((n_samples,),
                                                       dtype=np.intp)
        cdef SIZE_t* out_ptr = <SIZE_t*> out.data

        # Initialize auxiliary data-structure
        cdef DTYPE_t feature_value = 0
        cdef Node* node = NULL
        cdef DTYPE_t* X_sample = NULL
        cdef SIZE_t i = 0
        cdef SIZE_t k = 0

        # feature_to_sample as a data structure records the last seen sample
        # for each feature; functionally, it is an efficient way to identify
        # which features are nonzero in the present sample.
        cdef SIZE_t* feature_to_sample = NULL

        safe_realloc(&X_sample, n_features)
        safe_realloc(&feature_to_sample, n_features)

        with nogil:
            memset(feature_to_sample, -1, n_features * sizeof(SIZE_t))

            for i in range(n_samples):
                node = self.nodes

                for k in range(X_indptr[i], X_indptr[i + 1]):
                    feature_to_sample[X_indices[k]] = i
                    X_sample[X_indices[k]] = X_data[k]

                # While node not a leaf
                while node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    if feature_to_sample[node.feature] == i:
                        feature_value = <DTYPE_t>(X_sample[node.feature])

                    else:
                        feature_value = 0

                    if feature_value <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                out_ptr[i] = <SIZE_t>(node - self.nodes)  # node offset

            # Free auxiliary arrays
            free(X_sample)
            free(feature_to_sample)

        return out

    cpdef object decision_path(self, object X):
        """Finds the decision path (=node) for each sample in X."""
        if issparse(X):
            return self._decision_path_sparse_csr(X)
        else:
            return self._decision_path_dense(X)

    cdef inline object _decision_path_dense(self, object X):
        """Finds the decision path (=node) for each sample in X."""

        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef const DTYPE_t[:, :] X_ndarray = X
        cdef SIZE_t n_samples = X.shape[0]

        # Initialize output
        cdef np.ndarray[SIZE_t] indptr = np.zeros(n_samples + 1, dtype=np.intp)
        cdef SIZE_t* indptr_ptr = <SIZE_t*> indptr.data

        cdef np.ndarray[SIZE_t] indices = np.zeros(n_samples *
                                                   (1 + self.max_depth),
                                                   dtype=np.intp)
        cdef SIZE_t* indices_ptr = <SIZE_t*> indices.data

        # Initialize auxiliary data-structure
        cdef Node* node = NULL
        cdef SIZE_t i = 0

        with nogil:
            for i in range(n_samples):
                node = self.nodes
                indptr_ptr[i + 1] = indptr_ptr[i]

                # Add all external nodes
                while node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
                    indptr_ptr[i + 1] += 1

                    if X_ndarray[i, node.feature] <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                # Add the leave node
                indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
                indptr_ptr[i + 1] += 1

        indices = indices[:indptr[n_samples]]
        cdef np.ndarray[SIZE_t] data = np.ones(shape=len(indices),
                                               dtype=np.intp)
        out = csr_matrix((data, indices, indptr),
                         shape=(n_samples, self.node_count))

        return out

    cdef inline object _decision_path_sparse_csr(self, object X):
        """Finds the decision path (=node) for each sample in X."""

        # Check input
        if not isinstance(X, csr_matrix):
            raise ValueError("X should be in csr_matrix format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef np.ndarray[ndim=1, dtype=DTYPE_t] X_data_ndarray = X.data
        cdef np.ndarray[ndim=1, dtype=SIZE_t] X_indices_ndarray  = X.indices
        cdef np.ndarray[ndim=1, dtype=SIZE_t] X_indptr_ndarray  = X.indptr

        cdef DTYPE_t* X_data = <DTYPE_t*>X_data_ndarray.data
        cdef SIZE_t* X_indices = <SIZE_t*>X_indices_ndarray.data
        cdef SIZE_t* X_indptr = <SIZE_t*>X_indptr_ndarray.data

        cdef SIZE_t n_samples = X.shape[0]
        cdef SIZE_t n_features = X.shape[1]

        # Initialize output
        cdef np.ndarray[SIZE_t] indptr = np.zeros(n_samples + 1, dtype=np.intp)
        cdef SIZE_t* indptr_ptr = <SIZE_t*> indptr.data

        cdef np.ndarray[SIZE_t] indices = np.zeros(n_samples *
                                                   (1 + self.max_depth),
                                                   dtype=np.intp)
        cdef SIZE_t* indices_ptr = <SIZE_t*> indices.data

        # Initialize auxiliary data-structure
        cdef DTYPE_t feature_value = 0
        cdef Node* node = NULL
        cdef DTYPE_t* X_sample = NULL
        cdef SIZE_t i = 0
        cdef SIZE_t k = 0

        # feature_to_sample as a data structure records the last seen sample
        # for each feature; functionally, it is an efficient way to identify
        # which features are nonzero in the present sample.
        cdef SIZE_t* feature_to_sample = NULL

        safe_realloc(&X_sample, n_features)
        safe_realloc(&feature_to_sample, n_features)

        with nogil:
            memset(feature_to_sample, -1, n_features * sizeof(SIZE_t))

            for i in range(n_samples):
                node = self.nodes
                indptr_ptr[i + 1] = indptr_ptr[i]

                for k in range(X_indptr[i], X_indptr[i + 1]):
                    feature_to_sample[X_indices[k]] = i
                    X_sample[X_indices[k]] = X_data[k]

                # While node not a leaf
                while node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:

                    indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
                    indptr_ptr[i + 1] += 1

                    if feature_to_sample[node.feature] == i:
                        feature_value = <DTYPE_t>(X_sample[node.feature])

                    else:
                        feature_value = 0

                    if feature_value <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                # Add the leave node
                indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
                indptr_ptr[i + 1] += 1

            # Free auxiliary arrays
            free(X_sample)
            free(feature_to_sample)

        indices = indices[:indptr[n_samples]]
        cdef np.ndarray[SIZE_t] data = np.ones(shape=len(indices),
                                               dtype=np.intp)
        out = csr_matrix((data, indices, indptr),
                         shape=(n_samples, self.node_count))

        return out

    cdef np.ndarray _get_value_ndarray(self):
        """Wraps value as a 3-d NumPy array.

        The array keeps a reference to this Tree, which manages the underlying
        memory.
        """
        cdef np.npy_intp shape[3]
        shape[0] = <np.npy_intp> self.leaf_node_count
        shape[1] = <np.npy_intp> self.n_outputs
        shape[2] = <np.npy_intp> self.max_n_classes

        cdef np.ndarray arr
        arr = np.PyArray_SimpleNewFromData(3, shape, np.NPY_DOUBLE, self.value)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr

    cdef np.ndarray _get_node_ndarray(self):
        """Wraps nodes as a NumPy struct array.

        The array keeps a reference to this Tree, which manages the underlying
        memory. Individual fields are publicly accessible as properties of the
        Tree.
        """
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.internal_node_count
        cdef np.npy_intp strides[1]
        strides[0] = sizeof(Node)
        cdef np.ndarray arr
        Py_INCREF(NODE_DTYPE)
        arr = PyArray_NewFromDescr(<PyTypeObject *> np.ndarray,
                                   <np.dtype> NODE_DTYPE, 1, shape,
                                   strides, <void*> self.nodes,
                                   np.NPY_DEFAULT, None)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr
