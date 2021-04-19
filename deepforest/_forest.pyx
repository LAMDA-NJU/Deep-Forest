# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: language_level=3


cimport cython
import numpy as np
cimport numpy as np
np.import_array()

from numpy import uint8 as DTYPE
from numpy import float64 as DOUBLE

from .tree._tree cimport DTYPE_t
from .tree._tree cimport DOUBLE_t
from .tree._tree cimport SIZE_t

cdef SIZE_t _TREE_LEAF = -1


cpdef np.ndarray predict(object data,
                         const SIZE_t [:] feature,
                         const DTYPE_t [:] threshold,
                         const SIZE_t [:, ::1] children,
                         np.ndarray[DOUBLE_t, ndim=2] value):
    """Predict the class distributions or values for samples in ``data``.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_features)
        The numerical data to predict.
    feature : ndarray of shape (n_internal_nodes,)
        Stores the splitting feature for all internal nodes in the forest.
    threshold : ndarray of shape (n_internal_nodes,)
        Store the splitting threshold for all internal nodes in the forest.
    children : ndarray of shape (n_internal_nodes, 2)
        Store the IDs of left and right child for all internal nodes in the
        forest. Negative values indicate that the corresponding node is a
        leaf node.
    value : ndarray of shape (n_leaf_nodes, n_outputs)
        Store the prediction for all leaf nodes in the forest. The layout of
        ``children`` should be C-aligned. It is declared as ``np.ndarray``
        instead f typed memoryview to support splicing.

    Returns
    -------
    out : ndarray of shape (n_samples, n_outputs)
        The predicted class probabilities or values.
    """
    cdef:
        SIZE_t n_samples = data.shape[0]
        SIZE_t n_outputs = value.shape[1]
        SIZE_t n_indices
        np.ndarray[SIZE_t, ndim=1] indice = np.empty((n_samples,),
                                                     dtype=np.int32)
        np.ndarray[DOUBLE_t, ndim=2] out = np.zeros((n_samples, n_outputs),
                                                    dtype=DOUBLE)

    if not value.flags["C_CONTIGUOUS"]:
        value = np.ascontiguousarray(value)

    _apply_region(data, feature, threshold, children, indice)
    out += value.take(indice, axis=0, mode='clip')

    return out


cdef void _apply_region(const DTYPE_t [:, :] data,
                        const SIZE_t [:] feature,
                        const DTYPE_t [:] threshold,
                        const SIZE_t [:, ::1] children,
                        SIZE_t [:] out):
    """
    Find the terminal region (i.e., leaf node ID) for each sample in ``data``.
    """
    cdef:
        SIZE_t n_samples = data.shape[0]
        SIZE_t n_internal_nodes = feature.shape[0]
        SIZE_t i
        SIZE_t node_id
        SIZE_t node_feature
        DTYPE_t node_threshold
        SIZE_t left_child
        SIZE_t right_child

    with nogil:
        for i in range(n_samples):

            # Skip the corner case where the root node is a leaf node
            if n_internal_nodes == 0:
                out[i] = 0
                continue

            node_id = 0
            node_feature = feature[node_id]
            node_threshold = threshold[node_id]
            left_child = children[node_id, 0]
            right_child = children[node_id, 1]

            # While one of the two child of the current node is not a leaf node
            while left_child > 0 or right_child > 0:

                # If the left child is a leaf node
                if left_child <= 0:

                    # If X[sample_id] should be assigned to the left child
                    if data[i, node_feature] <= node_threshold:
                        out[i] = <SIZE_t>(_TREE_LEAF * left_child)
                        break
                    else:
                        node_id = right_child
                        node_feature = feature[node_id]
                        node_threshold = threshold[node_id]
                        left_child = children[node_id, 0]
                        right_child = children[node_id, 1]

                # If the right child is a leaf node
                elif right_child <= 0:

                    # If X[sample_id] should be assigned to the right child
                    if data[i, node_feature] > node_threshold:
                        out[i] = <SIZE_t>(_TREE_LEAF * right_child)
                        break
                    else:
                        node_id = left_child
                        node_feature = feature[node_id]
                        node_threshold = threshold[node_id]
                        left_child = children[node_id, 0]
                        right_child = children[node_id, 1]

                # If the left and right child are both internal nodes
                else:
                    if data[i, node_feature] <= node_threshold:
                        node_id = left_child
                    else:
                        node_id = right_child

                    node_feature = feature[node_id]
                    node_threshold = threshold[node_id]
                    left_child = children[node_id, 0]
                    right_child = children[node_id, 1]

            # If the left and child child are both leaf nodes
            if left_child <= 0 and right_child <= 0:
                if data[i, node_feature] <= node_threshold:
                    out[i] = <SIZE_t>(_TREE_LEAF * left_child)
                else:
                    out[i] = <SIZE_t>(_TREE_LEAF * right_child)
