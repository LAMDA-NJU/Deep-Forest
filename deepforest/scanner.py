"""
Implement methods on extracting structured data such as images or sequences.
"""


class ImageScanner(object):

    def __init__(self, kernel_size, stride=1, padding=0, backend="numpy"):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.backend = backend

    def _validate_parameters(self, image_size):
        pass

    def _torch_transform(self, X):
        pass

    def _numpy_transform(self, X):
        pass
    
    def fit_transform(self, X):
        self._validate_parameters(X.shape)

        if self.backend == "numpy":
            pass
        elif self.backend == "torch":
            pass

    def transform(self, X):
        pass
