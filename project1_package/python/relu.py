import numpy as np

def relu_forward(input_data):
    output = {
        'height': input_data['height'],
        'width': input_data['width'],
        'channel': input_data['channel'],
        'batch_size': input_data['batch_size'],
    }

    ###### Fill in the code here ######
    # Replace the following line with your implementation.
    # output['data'] = np.zeros_like(input_data['data'])
    
    # f(x) = max(0, x)
    output['data'] = np.maximum(0, input_data['data'])
    
    return output

def relu_backward(output, input_data, layer):
    ###### Fill in the code here ######
    # Replace the following line with your implementation.

    # output['diff'] -> gradients
    # Gradient of ReLU is 1 for input > 0, else 0
    input_od = output['diff'] * (input_data['data'] > 0).astype(float)

    return input_od
