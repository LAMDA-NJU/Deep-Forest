"""
Implement methods on extracting structured data such as images or sequences.
"""

import numpy as np


class ImageScanner(object):
    """
    Image scanner used to extract local patches. By using the image scanner
    to pre-process the image datasets, the performance of decision-tree based
    ensemble, including but not limited to the deep forest, is expected to
    improve.

    Parameters
    ----------
    kernel_size : :obj:`int` or :obj:`tuple`
      The size of the sliding blocks.
    stride : :obj:`int` or :obj:`tuple`, default=1
      The stride of the sliding blocks in the input spatial dimensions.
    padding : :obj:`int` or :obj:`tuple`, default=0
      Implicit zero padding to be added on both sides of input.
    backend : :obj:`{"numpy", "torch"}`, default="numpy"
      The backend used to extract image patches.
    channels_first : :obj:`bool`, default=True
      Whether the channel dimension is ahead of the dimension on the height
      and width of the image.
    use_gpu: :obj:`bool`, default=True
      Whether the ImageScanner uses cpu mode or gpu mode.
    """

    def __init__(
        self,
        kernel_size,
        stride=1,
        padding=0,
        backend="numpy",
        channels_first=True,
        use_gpu=False,
    ):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.backend = backend
        self.channels_first = channels_first
        self.use_gpu = use_gpu

    def _check_input(self, X):
        """Check the shape of input X."""
        if not isinstance(X, np.ndarray):
            err_msg = (
                "The input to `fit_transform` should be numpy.ndarray"
                "but got {} instead."
            )
            raise ValueError(err_msg.format(type(X)))

        if len(X.shape) != 4:
            shape_repr = "(num_samples, {}height, width{})".format(
                "channels, " if self.channels_first else "",
                "" if self.channels_first else ", channels",
            )
            msg = "ImageScanner now only supports 4-dim data of shape: {}"
            raise ValueError(msg.format(shape_repr))

        # Set attributes for the scanner
        if self.channels_first:
            _, self.n_channels, self.height, self.width = X.shape
        else:
            _, self.height, self.width, self.n_channels = X.shape

    def _torch_transform(self, X):
        """
        Generate image patches using the PyTorch backend.

        Parameters
        ----------
        X: :obj:`torch.Tensor`
          The input image datasets, the shape should be ``(n_samples,
          n_channels, height, width)``.

        Returns
        -------
        patches : obj:`numpy.ndarray
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

        # Swap dims if not channels first
        if not self.channels_first:
            X = X.permute(0, 3, 1, 2)

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
        if not self.channels_first:
            patches = patches.permute(0, 2, 3, 1)

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
        X: :obj:`numpy.ndarray`
          The input image datasets, the shape should be ``(n_samples,
          n_channels, height, width)``.
        """
        pass

    def fit_transform(self, X):
        """
        Set attributes for the image scanner and generate image patches using
        the Numpy or PyTorch backend.

        Parameters
        ----------
        X: :obj:`numpy.ndarray`
          The input image datasets of the shape ``(N, H, W)``,
          ``(N, C, H, W)``, or ``(N, H, W, C)``, where ``N`` is the number of
          images, ``C`` is the number of color channels, ``H`` is the height
          of image, and ``W`` is the width of image.

        Returns
        -------
        patches : obj:`numpy.ndarray
          The generated image patches.
        """
        self._check_input(X)

        # Numpy backend
        if self.backend == "numpy":
            return self._numpy_transform(X)
        # PyTorch backend
        elif self.backend == "torch":
            return self._torch_transform(X)
        else:
            err_msg = (
                "The name of the backend should be one of"
                " {{numpy, torch}}, but got {} instead."
            )
            raise NotImplementedError(err_msg.format(self.backend))

    def transform(self, X):
        """
        Generate image patches using the fitted image scanner.

        Parameters
        ----------
        X: :obj:`numpy.ndarray`
          The input image datasets of the shape ``(N, H, W)``,
          ``(N, C, H, W)``, or ``(N, H, W, C)``, where ``N`` is the number of
          images, ``C`` is the number of color channels, ``H`` is the height
          of image, and ``W`` is the width of image.

        Returns
        -------
        patches : obj:`numpy.ndarray
          The generated image patches.
        """
        # Check the consistency between X and the training data on scanner.
        if not (
            self.channels_first
            and X.shape
            == (X.shape[0], self.n_channels, self.height, self.width)
            or (
                (not self.channels_first)
                and X.shape
                == (X.shape[0], self.height, self.width, self.n_channels)
            )
        ):
            err_msg = (
                "Invalid input to `transform`. Pleased check whether"
                " it is same as the training data on scanner."
            )
            raise ValueError(err_msg)

        # Numpy backend
        if self.backend == "numpy":
            return self._numpy_transform(X)
        # PyTorch backend
        elif self.backend == "torch":
            return self._torch_transform(X)
