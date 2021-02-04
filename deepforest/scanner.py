"""
Implement methods on extracting structured data such as images or sequences.
"""

try:
    torch = __import__("torch")
except ModuleNotFoundError:
    msg = "Can not load the module torch when building the ImageScanner. \
           Please make sure that torch is installed."
    raise ModuleNotFoundError(msg)

try:
    np = __import__("numpy")
except ModuleNotFoundError:
    msg = "Can not load the module numpy when building the ImageScanner. \
           Please make sure that numpy is installed."
    raise ModuleNotFoundError(msg)

import torch.nn as nn


class ImageScanner(object):

    def __init__(self, kernel_size, stride=1, padding=0, backend="numpy",
                 channels_first=True):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.backend = backend
        self.channels_first = channels_first

    def _validate_parameters(self, image_size):
        pass

    def _torch_transform(self, X):
        """
        X: torch.Tensor with shape of (n, c, h, w) or (n, h, w, c)
        """

        if not self.channels_first:
            X = X.permute(0, 3, 1, 2)

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

        length = patch_sz * patch_sz * c
        if not self.channels_first:
            patches = patches.permute(0, 2, 3, 1)
        return patches.contiguous().view(-1, length).numpy()

    def _numpy_transform(self, X):
        pass

    def fit_transform(self, X):

        self._validate_parameters(X.shape)

        if not isinstance(X, np.ndarray):
            err_msg = "_torch_transform only accepts numpy.ndarray as input, \
                       but got {}"
            raise ValueError(err_msg.format(type(X)))

        if X.dtype != np.uint8:
            msg = "Warning! ImageScanner will force to transform data \
                   type of {} to uint8."
            print(msg.format(X.dtype))
            X = X.astype(np.uint8)

        if len(X.shape) != 4:
            shape_repr = "(num_samples, {}height, width{})".format(
                            "channels, " if self.channels_first else "",
                            "" if self.channels_first else ", channels"
                         )
            msg = "ImageScanner now only supports 4-dim data of {}"
            raise ValueError(msg.format(shape_repr))

        if self.backend == "numpy":
            return self._numpy_transform(X)
        elif self.backend == "torch":
            return self._torch_transform(torch.from_numpy(X))
        else:
            err_msg = "backend {} is not supported!"
            raise ValueError(err_msg.format(self.backend))

    def transform(self, X):
        pass
