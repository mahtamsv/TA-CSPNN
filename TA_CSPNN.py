"""
 TA_CPNN implementations 

Note1: 'image_data_format' should be 'channels_first'
Note2: We tested this using Tensorflow 1.12.0

"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Lambda
from tensorflow.keras.layers import DepthwiseConv2D, BatchNormalization
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K

K.set_image_data_format('channels_first')



def TA_CSPNN(nb_classes, Channels = 64, Timesamples = 90, 
             dropOut = 0.25,  timeKernelLen = 50, Ft = 11, Fs = 6):
    
    # Input shape is (trials, 1, number of channels, number of time samples)

    input_e      = Input(shape = (1, Channels, Timesamples))    
    convL1       = Conv2D(Ft, (1, timeKernelLen), padding = 'same',input_shape = (1, Channels, Timesamples), use_bias = False)(input_e)

    bNorm1       = BatchNormalization(axis = 1)(convL1)

    convL2       = DepthwiseConv2D((Channels, 1), use_bias = False, 
                                   depth_multiplier = Fs, depthwise_constraint = max_norm(1.))(bNorm1)  
    bNorm2       = BatchNormalization(axis = 1)(convL2)
    
    lambdaL      = Lambda(lambda x:x**2)(bNorm2)
    aPool        = AveragePooling2D((1, Timesamples))(lambdaL)

    dOutL       = Dropout(dropOut)(aPool)
        
    flatten      = Flatten(name = 'flatten')(dOutL)

    dense        = Dense(nb_classes, name = 'dense')(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs=input_e, outputs=softmax)
