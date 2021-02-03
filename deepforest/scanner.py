"""
Implement methods on extracting structured data such as images or sequences.
"""
import torch
import torch.nn as nn


class ImageScanner(object):

    def __init__(self, kernel_size, stride=1, padding=0, backend="numpy"):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.backend = backend

    def _validate_parameters(self, image_size):
        pass

    def _torch_transform(self, X):
        """the shape of X is n, c, h, w"""
        err_msg = "_torch_transform only accepts torch.Tensor as input, \
                  but got {}"
        assert isinstance(X, torch.Tensor), err_msg.format(type(X))

        n, c, _, _ = X.shape
        padding_wrapper = nn.ZeroPad2d(self.padding)
        padding_X = padding_wrapper(X)
        patches = padding_X.unfold(dimension=2, size=self.kernel_size,
                                   step=self.stride)
        patches = patches.unfold(dimension=3, size=self.kernel_size,
                                 step=self.stride).contiguous()
        patch_sz = self.kernel_size
        patches = patches.view(n, c, -1, patch_sz, patch_sz)
        patches = patches.permute(0, 2, 1, 3, 4)
        patches = patches.view(-1, c, patch_sz, patch_sz)
        return patches

    def _numpy_transform(self, X):
        pass

    def fit_transform(self, X):
        self._validate_parameters(X.shape)

        if self.backend == "numpy":
            return self._numpy_transform(X)
        elif self.backend == "torch":
            return self._torch_transform(X)
        else:
            err_msg = "backend {} is not supported!"
            raise ValueError(err_msg.format(self.backend))

    def transform(self, X):
        pass
