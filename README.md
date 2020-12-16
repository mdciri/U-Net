# U-Net

This is my version of the [U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) for image segmentation in 2D and 3D.

Using such code you can create a U-Net as deep as you like, at least until the feature filters shape allow a division by a factor 2.

The only differences from the Olaf Ronneberger one is that the down and up-sampling are done by a 2D (or 3D) convolution and 2D (or 3D) transopose convolution with stride 2, and evey convolution layer considers *same* padding.
