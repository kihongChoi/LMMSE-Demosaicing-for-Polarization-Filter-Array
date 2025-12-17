"""Block processing function similar to MATLAB's blockproc."""

import numpy as np
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp


def blockproc(image, block_size, fun, border_size=(0, 0), trim_border=True, use_parallel=True):
    """
    Process image in blocks, similar to MATLAB's blockproc.

    Args:
        image: Input image array
        block_size: Tuple (height, width) of block size
        fun: Function to apply to each block
        border_size: Tuple (vertical, horizontal) border size
        trim_border: Whether to trim border after processing
        use_parallel: Whether to use parallel processing

    Returns:
        Processed image
    """
    rows, cols = image.shape[:2]
    block_h, block_w = block_size
    border_v, border_h = border_size

    # Calculate number of blocks
    n_blocks_v = int(np.ceil(rows / block_h))
    n_blocks_h = int(np.ceil(cols / block_w))

    # Pad image with borders
    padded_image = np.pad(image, 
                         ((border_v, border_v), (border_h, border_h)) + ((0, 0),) * (len(image.shape) - 2),
                         mode='reflect')

    # Process each block
    def process_block(i, j):
        start_i = i * block_h
        start_j = j * block_w
        end_i = start_i + block_h + 2 * border_v
        end_j = start_j + block_w + 2 * border_h

        block = padded_image[start_i:end_i, start_j:end_j]
        result = fun(block)
        return i, j, result

    # Collect all block coordinates
    blocks = [(i, j) for i in range(n_blocks_v) for j in range(n_blocks_h)]

    if use_parallel:
        with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
            results = list(executor.map(lambda b: process_block(b[0], b[1]), blocks))
    else:
        results = [process_block(i, j) for i, j in blocks]

    # Determine output shape from first result
    _, _, first_result = results[0]
    if isinstance(first_result, np.ndarray):
        result_h, result_w = first_result.shape[:2]
        if len(first_result.shape) > 2:
            output_shape = (n_blocks_v * result_h, n_blocks_h * result_w) + first_result.shape[2:]
        else:
            output_shape = (n_blocks_v * result_h, n_blocks_h * result_w)
    else:
        result_h, result_w = block_h, block_w
        output_shape = (n_blocks_v * block_h, n_blocks_h * block_w)
        first_result = np.array(first_result)

    # Assemble output
    output = np.zeros(output_shape, dtype=first_result.dtype if isinstance(first_result, np.ndarray) else np.float64)

    for i, j, result in results:
        start_i = i * result_h
        start_j = j * result_w
        if isinstance(result, np.ndarray):
            end_i = start_i + result.shape[0]
            end_j = start_j + result.shape[1]
            if len(result.shape) == 2:
                output[start_i:end_i, start_j:end_j] = result
            elif len(result.shape) == 3:
                output[start_i:end_i, start_j:end_j, :] = result
            else:
                # Handle higher dimensions
                output[start_i:end_i, start_j:end_j] = result.reshape(result.shape[0], result.shape[1], -1)
        else:
            output[start_i:start_i + block_h, start_j:start_j + block_w] = result

    # Trim to original size if needed
    if trim_border:
        output = output[:rows, :cols]

    return output
