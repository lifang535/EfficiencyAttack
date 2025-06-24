
def input_smoothing(x, width=3, height=-1, kernel_size=None):
    """
    Simple input smoothing defense using median filtering
    
    Parameters:
        x: NumPy array input, with pixel values between 0-1
        width: Width of the sliding window (number of pixels)
        height: Height of the window. Same as width by default
        
    Returns:
        Smoothed array
    """
    # Ensure x is float32
    x = x.astype(np.float32)
    
    # Set default height if not specified
    if height == -1:
        height = width
    
    # Determine dimensions
    ndim = x.ndim
    
    if ndim == 2:  # Single channel 2D image
        # For 2D array, shape is (H, W)
        smoothed = ndimage.filters.median_filter(x, size=(width, height), mode='reflect')
    elif ndim == 3:  # Multi-channel 2D image or single-channel 3D image
        if x.shape[2] <= 4:  # Assuming RGB or RGBA image with shape (H, W, C)
            smoothed = np.zeros_like(x)
            for c in range(x.shape[2]):
                smoothed[:, :, c] = ndimage.filters.median_filter(x[:, :, c], size=(width, height), mode='reflect')
        else:  # Possibly 3D data
            smoothed = ndimage.filters.median_filter(x, size=(width, height, width), mode='reflect')
    elif ndim == 4:  # Batch of images with shape (B, H, W, C)
        smoothed = np.zeros_like(x)
        for b in range(x.shape[0]):
            for c in range(x.shape[3]):
                # Using the exact format from median_filter_py
                temp_input = x[b:b+1, :, :, c:c+1]  # Shape becomes (1, H, W, 1)
                temp_output = ndimage.filters.median_filter(temp_input, size=(1, width, height, 1), mode='reflect')
                smoothed[b, :, :, c] = temp_output[0, :, :, 0]
    else:
        # For other dimensions, use a general approach
        filter_size = tuple([width if i < ndim else 1 for i in range(ndim)])
        smoothed = ndimage.filters.median_filter(x, size=filter_size, mode='reflect')
    
    # Ensure values remain in 0-1 range
    smoothed = np.clip(smoothed, 0, 1)
    return smoothed