import numpy as np


def _im2col_indices(X, kernel_size, stride=1, padding=1):
    """
    Generate image patches with fancy indexing, modified from the cs231n
    course on computer vision.
    """

    def get_im2col_indices(shape, kernel_size, stride=1, padding=0):
        N, C, H, W = shape
        assert (H + 2 * padding - kernel_size) % stride == 0
        assert (W + 2 * padding - kernel_size) % stride == 0
        out_height = (H + 2 * padding - kernel_size) // stride + 1
        out_width = (W + 2 * padding - kernel_size) // stride + 1

        i0 = np.repeat(np.arange(kernel_size), kernel_size)
        i0 = np.tile(i0, C)
        i1 = stride * np.repeat(np.arange(out_height), out_width)
        j0 = np.tile(np.arange(kernel_size), kernel_size * C)
        j1 = stride * np.tile(np.arange(out_width), out_height)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        k = np.repeat(np.arange(C), kernel_size * kernel_size).reshape(-1, 1)

        return k, i, j

    padding_X = np.pad(
        X,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant",
    )

    k, i, j = get_im2col_indices(X.shape, kernel_size, stride, padding)

    cols = padding_X[:, k, i, j]
    C = X.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(kernel_size * kernel_size * C, -1)

    return cols


class ImageScanner(object):
    """
    Image scanner used to extract local patches. By using the image scanner
    to pre-process the image datasets, the performance of decision-tree based
    ensemble, including but not limited to the deep forest, is expected to
    improve.

    Parameters
    ----------
    kernel_size : :obj:`int`
      The size of the sliding blocks.
    stride : :obj:`int`, default=1
      The stride of the sliding blocks in the input spatial dimensions.
    padding : :obj:`int`, default=0
      Implicit zero padding to be added on both sides of input.
    backend : :obj:`{"numpy", "torch"}`, default="numpy"
      The backend used to extract image patches.
    use_gpu: :obj:`bool`, default=True
      Whether the ImageScanner uses cpu mode or gpu mode. This parameter has
      no effect when using the ``numpy`` backend.
    """

    def __init__(
        self,
        kernel_size,
        stride=1,
        padding=0,
        backend="numpy",
        use_gpu=False,
    ):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.backend = backend
        self.use_gpu = use_gpu

    def _check_input(self, X):
        """Check the shape of input X."""
        if not isinstance(X, np.ndarray):
            err_msg = (
                "The input to `fit_transform` should be numpy.ndarray"
                "but got {} instead."
            )
            raise ValueError(err_msg.format(type(X)))

        # Add the channel dimension if missing
        if len(X.shape) == 3:
            X = np.expand_dims(X, 1)

        if len(X.shape) != 4:
            msg = (
                "ImageScanner now only supports 4-dim data of shape:"
                " (n_samples, n_channels, height, width)."
            )
            raise ValueError(msg)

        # Set attributes for the scanner
        _, self.n_channels, self.height, self.width = X.shape

        return X

    def _torch_transform(self, X):
        """
        Generate image patches using the PyTorch backend.

        Parameters
        ----------
        X : :obj:`numpy.ndarray`
          The input image datasets, the shape should be ``(n_samples,
          n_channels, height, width)``.

        Returns
        -------
        patches : obj:`numpy.ndarray`
          The generated image patches.
        """
        try:
            torch = __import__("torch")
        except ModuleNotFoundError:
            msg = (
                "Cannot load the module torch when building the "
                "ImageScanner. Please make sure that torch is"
                " installed."
            )
            raise ModuleNotFoundError(msg)

        X = torch.from_numpy(X)  # numpy.ndarry -> torch.tensor

        if self.use_gpu:
            # Move tensor to GPU memory
            X = X.cuda()

        # Padding with 0
        padding_wrapper = torch.nn.ZeroPad2d(self.padding)
        padding_X = padding_wrapper(X)

        # Generate patches
        n_samples, n_channels, _, _ = padding_X.shape
        patches = padding_X.unfold(
            dimension=2, size=self.kernel_size, step=self.stride
        )
        patches = patches.unfold(
            dimension=3, size=self.kernel_size, step=self.stride
        ).contiguous()
        patch_sz = self.kernel_size
        patches = patches.view(n_samples, n_channels, -1, patch_sz, patch_sz)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        patches = patches.view(-1, n_channels, patch_sz, patch_sz)

        length = patch_sz * patch_sz * n_channels  # patch size: c * W * W

        if self.use_gpu:
            torch.cuda.synchronize()
            # Move back
            patches = patches.cpu()

        # Squeeze
        return patches.contiguous().view(-1, length).numpy()

    def _numpy_transform(self, X):
        """
        Generate image patches using the Numpy backend.

        Parameters
        ----------
        X : :obj:`numpy.ndarray`
          The input image datasets, the shape should be ``(n_samples,
          n_channels, height, width)``.

        Returns
        -------
        patches : obj:`numpy.ndarray`
          The generated image patches.
        """
        patches = _im2col_indices(
            X, self.kernel_size, self.stride, self.padding
        )

        return np.transpose(patches, (1, 0))

    def fit_transform(self, X, y):
        """
        Set attributes for the image scanner and generate image patches using
        the Numpy or PyTorch backend.

        Parameters
        ----------
        X : :obj:`numpy.ndarray`
          The input image datasets of the shape ``(N, H, W)`` or
          ``(N, C, H, W)``, where ``N`` is the number of images, ``C`` is the
          number of color channels, ``H`` is the height of image, and ``W``
          is the width of image.

        y : :obj:`numpy.ndarray` of shape (n_samples,)
          The class labels of input samples.

        Returns
        -------
        patches : :obj:`numpy.ndarray`
          The generated image patches.
        """
        X = self._check_input(X)

        # Numpy backend
        if self.backend == "numpy":
            X_trans = self._numpy_transform(X)
        # PyTorch backend
        elif self.backend == "torch":
            X_trans = self._torch_transform(X)
        else:
            err_msg = (
                "The name of the backend should be one of"
                " {{numpy, torch}}, but got {} instead."
            )
            raise NotImplementedError(err_msg.format(self.backend))

        # Label transformation
        n_patches_per_image = (
            (self.height + 2 * self.padding - self.kernel_size) // self.stride
            + 1
        ) * (
            (self.width + 2 * self.padding - self.kernel_size) // self.stride
            + 1
        )
        y_trans = np.repeat(y.reshape(-1, 1), n_patches_per_image, axis=0)

        return X_trans, y_trans.reshape(-1)

    def transform(self, X):
        """
        Generate image patches using the fitted image scanner.

        Parameters
        ----------
        X : :obj:`numpy.ndarray`
          The input image datasets of the shape ``(N, H, W)`` or
          ``(N, C, H, W)``, where ``N`` is the number of images, ``C`` is the
          number of color channels, ``H`` is the height of image, and ``W``
          is the width of image.

        Returns
        -------
        patches : :obj:`numpy.ndarray`
          The generated image patches.
        """
        # Check the consistency between X and the training data on scanner.
        if not (
            X.shape == (X.shape[0], self.n_channels, self.height, self.width)
        ):
            err_msg = (
                "Invalid input to `transform`. Pleased check whether"
                " it is same as the training data on scanner."
            )
            raise ValueError(err_msg)

        # Numpy backend
        if self.backend == "numpy":
            X_trans = self._numpy_transform(X)
        # PyTorch backend
        elif self.backend == "torch":
            X_trans = self._torch_transform(X)

        return X_trans
