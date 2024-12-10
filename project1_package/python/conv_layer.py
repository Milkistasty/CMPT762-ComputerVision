import numpy as np
from utils import im2col_conv, col2im_conv, im2col_conv_batch

def conv_layer_forward(input_data, layer, param):
    """
    Forward pass for a convolutional layer.

    Parameters:
    - input_data (dict): A dictionary containing the input data.
    - layer (dict): Layer configuration containing parameters such as kernel size, padding, stride, etc.
    - param (dict): A dictionary containing the parameters 'b' and 'w'.
    
    Returns:
    - output (dict): A dictionary containing the output of the convolution layer.
    """

    h_in = input_data['height']
    w_in = input_data['width']
    c = input_data['channel']
    batch_size = input_data['batch_size']
    k = layer['k']
    pad = layer['pad']
    stride = layer['stride']
    num = layer['num']

    # Resolve output shape
    h_out = (h_in + 2 * pad - k) // stride + 1
    w_out = (w_in + 2 * pad - k) // stride + 1

    assert h_out == int(h_out), 'h_out is not an integer'
    assert w_out == int(w_out), 'w_out is not an integer'

    input_n = {
        'height': h_in,
        'width': w_in,
        'channel': c,
        'data': input_data['data'],
        'batch_size': batch_size
    }

    output = {
        'height': h_out,
        'width': w_out,
        'channel': num,
        'batch_size': batch_size,
        'data': np.zeros((h_out * w_out * num, batch_size))  # replace 'data' value with your implementation
    }

    ############## Fill in the code here ###############
    # Hint: use im2col_conv_batch for faster computation

    col = im2col_conv_batch(input_n, layer, h_out, w_out)  # Shape: (k*k*c, h_out*w_out, batch_size)

    w = param['w']  # shape: (k*k*c, num)
    b = param['b']  # shape: (num,)

    # Convolution
    # col shape: (k*k*c, h_out*w_out, batch_size)
    # col.T shape: (batch_size, h_out*w_out, k*k*c)
    # col.T @ w = (batch_size, h_out*w_out, k*k*c) @ (k*k*c, num) = (batch_size, h_out*w_out, num)
    out = col.T @ w
    out = out.transpose(2, 1, 0)  # Shape: (num, h_out*w_out, batch_size)

    b = b.reshape(num, 1)  # Shape: (num, 1)
    out += b[:, np.newaxis, :]  # b shape: (num, 1, 1), out shape: (num, h_out*w_out, batch_size)

    output['data'] = out.reshape(-1, batch_size)  # Shape: (h_out*w_out*num, batch_size)

    return output


def conv_layer_backward(output, input_data, layer, param):
    """
    Compute the backward pass for the convolution layer.
    
    Parameters:
    - output (dict): A dictionary containing the output of the forward pass.
    - input_data (dict): A dictionary containing the original input to the forward function.
    - layer (dict): Layer configuration containing parameters such as kernel size, padding, stride, etc.
    - param (dict): A dictionary containing the parameters 'b' and 'w'.

    Returns:
    - param_grad (dict): A dictionary containing the gradients with respect to the parameters 'b' and 'w'.
    - input_od (numpy.ndarray): The gradients with respect to the input.
    """
    
    h_in = input_data['height']
    w_in = input_data['width']
    c = input_data['channel']
    batch_size = input_data['batch_size']
    k = layer['k']
    group = layer['group']
    num = layer['num']

    h_out = output['height']
    w_out = output['width']
    input_n = {'height': h_in, 'width': w_in, 'channel': c}
    
    input_od = np.zeros(input_data['data'].shape)
    param_grad = {'b': np.zeros(param['b'].shape), 'w': np.zeros(param['w'].shape)}

    for n in range(batch_size):
        input_n['data'] = input_data['data'][:, n]
        col = im2col_conv(input_n, layer, h_out, w_out)
        col = np.reshape(col, (k*k*c, h_out*w_out), order='F')
        col_diff = np.zeros(col.shape)
        temp_data_diff = np.reshape(output['diff'][:, n], (h_out*w_out, num), order='F')
        
        for g in range(group):
            g_c_idx = slice(g*k*k*c//group, (g+1)*k*k*c//group)
            g_num_idx = slice(g*num//group, (g+1)*num//group)
            col_g = col[g_c_idx, :]
            weight = param['w'][:, g_num_idx]
            
            # get the gradient of param
            param_grad['b'][:, g_num_idx] += np.sum(temp_data_diff[:, g_num_idx], axis=0)
            param_grad['w'][:, g_num_idx] += col_g.dot(temp_data_diff[:, g_num_idx])
            col_diff[g_c_idx, :] = weight.dot(temp_data_diff[:, g_num_idx].T)
            
        im = col2im_conv(col_diff.ravel(order='F'), input_data, layer, h_out, w_out)
        # set the gradient w.r.t to input.data
        input_od[:, n] = im.ravel(order='F')

    return param_grad, input_od



