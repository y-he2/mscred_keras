import numpy as np
import pandas as pd
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers

"""
	Input either the input tensor itself or its shape, the MSCRED model will be generated automatically according to the shape. 
	Input shape: (batches, scales, img_width, img_height, timesteps)
	Output shape: Infered from the input shape
"""
def build_mscred( 
    input_tensor = None, 
    input_shape = None, 
    stride_sizes = [1,2,2,2], 
    cnn_filter_sizes = [32,64,128,256]
): 
    if( input_shape is not None ): 
        input_tensor_shape = input_shape
    elif( input_tensor is not None ): 
        input_tensor_shape = input_tensor.shape
    else:
        raise ValueError( "Error: Input shape must be provided either as the tensor it self or shape tuple!" )
        
    if( len( stride_sizes ) != len( cnn_filter_sizes ) ):
        raise ValueError( "Error: stride_sizes and cnn_filter_sizes length mismatched!" )
        
    layer_input = keras.Input(
        shape = input_tensor_shape[ 1: ], 
        name = "input"
    ) 
    print( layer_input )

    MAX_DEPTH = len( stride_sizes )

    ## Generate CNN layers. 
    cnn_layers = []
    layer_conv = layer_input
    for ll in range( MAX_DEPTH ):
        layer_conv = keras.layers.TimeDistributed( 
            keras.layers.Conv2D( 
                filters = cnn_filter_sizes[ ll ], 
                kernel_size = 2, 
                strides = (stride_sizes[ ll ], stride_sizes[ ll ]),
                kernel_regularizer = keras.regularizers.l2( 0.0001 ), #OPTIMIZE
                kernel_initializer = 'lecun_normal',
                activation = 'selu',
                padding='same', 
                name = "time_dist_conv_" + str( ll ), 
            )
        )( layer_conv )
        print( layer_conv )
        cnn_layers.append( layer_conv )

    ## Generate ConvLSTM layers. 
    rnn_layers = []
    for ll in range( MAX_DEPTH ):
        layer_convlstm = keras.layers.ConvLSTM2D(
            filters = cnn_filter_sizes[ ll ], 
            kernel_size = 2,
        #     return_sequences = True, 
            kernel_regularizer = keras.regularizers.l2( 0.0001 ), #OPTIMIZE
            data_format = 'channels_last', 
            dropout = 0.2, 
            padding = 'same', 
            name = "conv_lstm_" + str( ll ) 
        )( cnn_layers[ ll ] )
        print( layer_convlstm )
        rnn_layers.append( layer_convlstm )


    ## Generate the reconstruction layers. 
    prev_deconv = None
    for ll in range( MAX_DEPTH - 1, -1, -1 ):
        ## Fix the exceeding row and col during deconv upsampling (x2). 
        if( prev_deconv is not None ): 
            if( prev_deconv.shape.as_list()[ 1 ] > rnn_layers[ min( ll, len( rnn_layers ) - 1 ) ].shape[ 1 ] ):
                prev_deconv = keras.layers.Lambda( 
                    lambda tt: tt[ :, :-1, :-1, : ], 
                    name = 'slice_deconv_' + str( ll ) 
                )( prev_deconv )
        if( prev_deconv is not None ): 
            layer_concat = keras.layers.Concatenate()( [rnn_layers[ ll ], prev_deconv] )
        else:
            layer_concat = rnn_layers[ ll ]
        deconv_filter_size = input_tensor.shape[ -1 ] if( ll == 0 ) else cnn_filter_sizes[ max( ll - 1, 0 ) ]
        layer_deconv = keras.layers.Conv2DTranspose(
            filters = deconv_filter_size, 
            kernel_size = 2, 
            strides = (stride_sizes[ ll ], stride_sizes[ ll ]),
            kernel_regularizer = keras.regularizers.l2( 0.0001 ), #OPTIMIZE
            kernel_initializer = 'lecun_normal',
            activation = 'selu', 
            padding = 'same', 
        )( layer_concat )

        prev_deconv = layer_deconv
        print( layer_deconv )
        
    layer_inputs = layer_input
    layer_outputs = layer_deconv

    model = keras.Model(
        inputs = layer_inputs,
        outputs = layer_outputs,
    )
    return( model )

