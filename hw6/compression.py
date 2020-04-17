import numpy as np


def compress_image(image, num_values):
    """Compress an image using SVD and keeping the top `num_values` singular values.

    Args:
        image: numpy array of shape (H, W)
        num_values: number of singular values to keep

    Returns:
        compressed_image: numpy array of shape (H, W) containing the compressed image
        compressed_size: size of the compressed image
    """
    compressed_image = None
    compressed_size = 0

    # YOUR CODE HERE
    # Steps:
    #     1. Get SVD of the image
    #     2. Only keep the top `num_values` singular values, and compute `compressed_image`
    #     3. Compute the compressed size
    U, S, VT = np.linalg.svd(image)

    for i in range(num_values):
    	U[:,i] = U[:,i] * S[i]

    compressed_image = np.matmul(U[:,0:num_values], VT[0:num_values,:])
    compressed_size = U[:,0:num_values].size + VT[0:num_values,:].size + S[0:num_values].size
    # END YOUR CODE

    assert compressed_image.shape == image.shape, \
           "Compressed image and original image don't have the same shape"

    assert compressed_size > 0, "Don't forget to compute compressed_size"

    return compressed_image, compressed_size

# a = np.array([1,2,3,4])
# print(a[0:2])