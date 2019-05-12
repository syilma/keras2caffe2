__author__ = "Samson Yilma"
__copyright__ = "Copyright 2019, Samson Yilma"


import os
import keras
import numpy as np
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, Reshape, AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, concatenate
from keras.layers import Softmax
  
import keras.layers
from caffe2.python import core, model_helper, workspace, brew
from caffe2.proto import caffe2_pb2

def save_net(INIT_NET, PREDICT_NET, model, input_shape) :
    from caffe2.python import utils
    
    with open(PREDICT_NET, 'wb') as f:
        f.write(model.net._net.SerializeToString())
    
    init_net = caffe2_pb2.NetDef()
    for param in model.params:
        blob = workspace.FetchBlob(param)
        shape = blob.shape
        op = core.CreateOperator("GivenTensorFill", [], [param],arg=[utils.MakeArgument("shape", shape), 
                                                                     utils.MakeArgument("values", blob)])
        init_net.op.extend([op])
        
    input_name = model.net.external_inputs[0]
    init_net.op.extend([core.CreateOperator("ConstantFill", [], [input_name], shape=input_shape)])
    
    with open(INIT_NET, 'wb') as f:
        f.write(init_net.SerializeToString())
        
def get_padding_sizes(input_s, kernel, stride ):
    import math

    pad_list = []
    
    for kk in range(len(input_s)):
        out_s = math.ceil(float(input_s[kk]) / float(stride[kk]))
        pad_s = (out_s - 1) * stride[kk] + kernel[kk] - input_s[kk]
        
        pad_1 = np.max([0, int(pad_s/2)])
        pad_2 = pad_s - pad_1
        
        pad_list.append((pad_1, pad_2) )
    
    return tuple(pad_list)

def create_caffe2_model(model, input_shape, use_cudnn=True, init_params=False, keras_channel_last=True): 

    arg_scope = {'order': 'NCHW', 'use_cudnn': use_cudnn}
    caffe2_model = model_helper.ModelHelper(name='model', init_params=init_params, arg_scope=arg_scope)

    num_conv_layers = 0
    
    layer_num = 0
    layer_sizes = {}
    prev_layer_name = ''
    
    for layer in model.layers:
        
        inb_node = layer._inbound_nodes[0]
        num_input_layers = len(inb_node.inbound_layers)
        
        input_name_list = []
        
        for ii in range(0, num_input_layers):
            inp_layer = inb_node.inbound_layers[ii]
           
            input_name_list.append(inp_layer.name)
            prev_layer_name = inp_layer.name
            
            if isinstance(inp_layer, keras.layers.Flatten):         
                pass
                #pinb_node = inp_layer._inbound_nodes[0] 
                #prev_layer_name = pinb_node.inbound_layers[0].name
                                  
        name = layer.name
        
        config = layer.get_config()
        inputShape = layer.input_shape
        outputShape = layer.output_shape
                                       
        if isinstance(layer, keras.engine.input_layer.InputLayer):
            input_sizes = (input_shape[2], input_shape[3])
            layer_sizes[name] = input_sizes       
        else:
            if (input_name_list[0] not in layer_sizes):
                raise ValueError("Can't find layer size for ", input_name_list[0] )
            else:
                input_sizes = layer_sizes[input_name_list[0]]
               
        layer_dim = len(outputShape)               
        if (layer_dim == 4):
            if (keras_channel_last):
                out_sizes = (outputShape[1], outputShape[2])
            else:
                out_sizes = (outputShape[2], outputShape[3])
        elif (layer_dim == 2):
            out_sizes = (0, 0) #flattened
        else:
            raise ValueError('Unsupported layer dimension : {0}'.format(layer_dim) )
                        
        if isinstance(layer, keras.layers.Flatten):
            tmp_prev = prev_layer_name

            if (keras_channel_last):
                tmp_prev = prev_layer_name + '_transpose'   #nb, img_h, img_w, chan <-- nb, chan, img_h, img_w
                c2_layer = brew.transpose(caffe2_model, prev_layer_name, tmp_prev, axes=(0, 2, 3, 1))
            
            c2_layer = caffe2_model.net.Flatten(
                tmp_prev,
                name
            )    
            
            #print('FLatten previous layer ', prev_layer_name, ' current layer ', name , 'inputshape ', inputShape) 
            
            layer_sizes[name] = out_sizes 
        
        elif isinstance(layer, keras.layers.Dropout):
            #print('name is ', name, ' prev_layer_name ', prev_layer_name)
            c2_layer = caffe2_model.net.Dropout(
                prev_layer_name,
                name,
                is_test = True
                #ratio=config['rate']
            )
                        
            #same size
            layer_sizes[name] = input_sizes
            
        elif (isinstance(layer, keras.layers.convolutional.Conv2D)):
            
            dim_in = inputShape[-1]
            dim_out = outputShape[-1]
            kernel = config['kernel_size'][0]
            stride = config['strides'][0]
              
            if (config['padding'] == 'same'):             
                pad_sizes = get_padding_sizes(input_sizes, config['kernel_size'], config['strides'])                      
            elif (config['padding'] == 'valid'):
                pad_sizes = ((0, 0), (0, 0))
            else:
                raise ValueError('unsupported padding')
            
            #print('pad sizes ', pad_sizes)
            
            layer_sizes[name] = out_sizes
                                                
            c2_layer = brew.conv(caffe2_model,
                      prev_layer_name,
                      name,
                      dim_in=dim_in,
                      dim_out=dim_out,
                      kernel=kernel,
                      stride=stride,
                      pad_l=pad_sizes[0][0], pad_r=pad_sizes[0][1], pad_t=pad_sizes[1][0], pad_b=pad_sizes[1][1]
                    )
            
            if config['activation'] == 'linear':
                pass
            elif config['activation'] == 'relu':
                c2_layer = brew.relu(caffe2_model, name, name)
            elif config['activation'] == 'softmax':
                #c2_layer = brew.softmax(caffe2_model, name, name)
                c2_layer = brew.softmax(caffe2_model, name, 'softmax')
            else:
                raise ValueError('The only supported activation for conv layer is relu')
                            
        elif isinstance(layer, keras.layers.MaxPooling2D):
            kernel = config['pool_size'][0]
            stride = config['strides'][0]
            
            pad_size = ((0, 0),(0,0))
            layer_sizes[name] = out_sizes
            
            c2_layer = brew.max_pool(caffe2_model,
                          prev_layer_name,
                          name,
                          kernel=kernel,
                          stride=stride)
                       
        elif isinstance(layer, keras.layers.AveragePooling2D):
            kernel = config['pool_size'][0]
            stride = config['strides'][0]
            
            pad_size = ((0, 0),(0,0))
            layer_sizes[name] = out_sizes
            
            c2_layer = brew.average_pool(caffe2_model,
                          prev_layer_name,
                          name,
                          kernel=kernel,
                          stride=stride)
                        
        elif isinstance(layer, keras.layers.BatchNormalization):

            dim_in = inputShape[-1]
            epsilon = config['epsilon']
            momentum = config['momentum']
            c2_layer = brew.spatial_bn(caffe2_model,
                            prev_layer_name,
                            name, 
                            dim_in=dim_in, 
                            epsilon=epsilon, 
                            momentum=momentum,
                            is_test=True)
             
            #same size
            layer_sizes[name] = input_sizes
                        
        elif (isinstance(layer, keras.layers.core.Dense)):
            
            dim_in = inputShape[-1]    
            dim_out = outputShape[-1]
               
            #print('input shape for dense is ', inputShape)
            if (len(inputShape) == 2):   #flattened input
                c2_layer = brew.fc(caffe2_model,
                        prev_layer_name,
                        name,
                        dim_in=dim_in,
                        dim_out=dim_out)
            else: #fully convolutional input
                c2_layer = brew.conv(caffe2_model,
                        prev_layer_name,
                        name,
                        dim_in=dim_in,
                        dim_out=dim_out,
                        kernel=1,
                        stride=1)

            activation = config['activation']
            if activation == 'relu':
                c2_layer = brew.relu(caffe2_model, name, name)
            elif activation == 'softmax':
                c2_layer = brew.softmax(caffe2_model, name, 'softmax')
            elif activation == 'linear':
                pass # 
            else:
                raise ValueError('The only supported activations for fc layer are relu and softmax')
                    
            #same size
            layer_sizes[name] = input_sizes
            
        elif (isinstance(layer, keras.layers.advanced_activations.LeakyReLU)):
                        
            dim_in = inputShape[-1]
            
            c2_layer = caffe2_model.net.LeakyRelu(
                prev_layer_name,
                name,
                alpha=config['alpha']
            )
                        
            #same size
            layer_sizes[name] = input_sizes
            
        elif (isinstance(layer, keras.layers.merge.Add)):
                  
            c2_layer = brew.sum(caffe2_model, [input_name_list[0], input_name_list[1]], name)
                        
            #same size
            layer_sizes[name] = input_sizes
        
                    
    layer_num = layer_num + 1
    if (layer_num == len(model.layers)):
        caffe2_model.net.AddExternalOutput(c2_layer)
        
    return caffe2_model

def set_weights(keras_model, caffe2_model):
    '''
    copies keras model weights to caffe2 model
    '''
    for layer in keras_model.layers:  
        name = layer.name 

        if isinstance(layer, keras.layers.Conv2D): 
            win = layer.get_weights()[0]
            w = layer.get_weights()[0].transpose((3, 2, 0, 1))
            b = layer.get_weights()[1]
            workspace.FeedBlob(name + '_w', w)
            workspace.FeedBlob(name + '_b', b)
                
        elif isinstance(layer, keras.layers.BatchNormalization):
            s = layer.get_weights()[0]
            b = layer.get_weights()[1]
            rm = layer.get_weights()[2]
            riv = layer.get_weights()[3]
            
            workspace.FeedBlob(name + '_s', s)
            workspace.FeedBlob(name + '_b', b)
            workspace.FeedBlob(name + '_rm', rm)
            workspace.FeedBlob(name + '_riv', riv)
            
            # Add rm and riv parameters of spatial_bn layers to params list of the model
            caffe2_model.params.append(workspace.StringifyBlobName(name + '_rm'))
            caffe2_model.params.append(workspace.StringifyBlobName(name + '_riv'))

        elif isinstance(layer, keras.layers.Dense):     
            w_keras = layer.get_weights()[0]
            b = layer.get_weights()[1]
            
            inputShape = layer.input_shape
            if (len(inputShape) == 2): #flattened input
                w = w_keras.transpose();
            elif (len(inputShape) == 4): #fully convolutional input
                wtemp1 = w_keras.reshape(1, 1, w_keras.shape[0], w_keras.shape[1])
                w = wtemp1.transpose((3, 2, 0, 1))
            else:
                 raise ValueError('unsupported size in Dense ')
                
            workspace.FeedBlob(name + '_w', w)
            workspace.FeedBlob(name + '_b', b)
            
        else:
            pass
            #print("weight not set for layer ", layer.name)

def caffe2_from_keras(keras_model, input_shape, use_cudnn=True, init_params=False, keras_channel_last=True):
    
    caffe2_model = create_caffe2_model(keras_model, input_shape, use_cudnn=use_cudnn, init_params=init_params, keras_channel_last=keras_channel_last)
    
    np_data = np.zeros(input_shape, dtype=np.float32)
    input_layer_name = keras_model.layers[0].get_config()['name']
    
    workspace.FeedBlob(input_layer_name, np_data)

    workspace.RunNetOnce(caffe2_model.param_init_net)
    workspace.CreateNet(caffe2_model.net)
    
    set_weights(keras_model, caffe2_model)
    
    return caffe2_model
    
def keras2caffe2(keras_model, input_w, input_h, init_net_filename, predict_net_filename, use_cudnn=True, keras_channel_last=True):
        
    input_shape = (1, 1, input_h, input_w) #caffe2 channel first convention
    
    #create training model
    train_model = caffe2_from_keras(keras_model, input_shape=input_shape, use_cudnn=use_cudnn, init_params=True, keras_channel_last=keras_channel_last)
    
    test_model = caffe2_from_keras(keras_model, input_shape=input_shape, use_cudnn=use_cudnn, init_params=False, keras_channel_last=keras_channel_last)

    workspace.RunNetOnce(test_model.param_init_net)
    workspace.CreateNet(test_model.net, overwrite=True)

    if (os.path.exists(os.path.dirname(init_net_filename)) and os.path.exists(os.path.dirname(predict_net_filename)) ):
        save_net(init_net_filename, predict_net_filename, test_model, input_shape=input_shape)
    
    return train_model, test_model
