import numpy as np


def inner_product_forward(input, layer, param):
    """
    Forward pass of inner product layer.

    Parameters:
    - input (dict): Contains the input data.
    - layer (dict): Contains the configuration for the inner product layer.
    - param (dict): Contains the weights and biases for the inner product layer.
    """

    # d: input dimension (height*width), k: batch size
    d, k = input["data"].shape 
    # n: number of neurons
    n = param["w"].shape[1]
    w = param['w']
    b = param['b']

    ###### Fill in the code here ######

    # Initialize output data structure
    output = {
        "height": n,
        "width": 1,
        "channel": 1,
        "batch_size": k,
        "data": w.T @ input['data'] + b.T # replace 'data' value with your implementation
    }

    return output


def inner_product_backward(output, input_data, layer, param):
    """
    Backward pass of inner product layer.

    Parameters:
    - output (dict): Contains the output data.
    - input_data (dict): Contains the input data.
    - layer (dict): Contains the configuration for the inner product layer.
    - param (dict): Contains the weights and biases for the inner product layer.
    """
    param_grad = {}
    ###### Fill in the code here ######
    # Replace the following lines with your implementation.
    
    # Gradient of loss with respect to biases
    param_grad['b'] = np.sum(output['diff'], axis=1, keepdims=True).T  # Shape: (1, n)
    # Gradient of loss with respect to weights
    param_grad['w'] = input_data['data'] @ output['diff'].T  # Shape: (d, n)
    # Gradient of loss with respect to the input data
    input_od = param['w'] @ output['diff']  # Shape: (d, k)

    return param_grad, input_od