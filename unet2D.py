import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Conv2DTranspose

class DoubleConvs(tf.keras.layers.Layer):
    ''' Double 2D convolution - BatchNorm - ReLU activation. 
    
        Layer initilizate with the same num of output filters, kernel size, and initilization type in both convolutions.
        
        input: a tensor of shape (batch_size, high, width, channels).
        output: a tensor of shape (batch_size, high, width, channels_out).
    '''
    def __init__(self, channels_out, ks):
        super(DoubleConvs, self).__init__(name='Double_Convs_Layer')
        
        self.conv1 = Conv2D(filters=channels_out, kernel_size=ks, padding='same', kernel_initializer='he_normal')
        self.norm1 = BatchNormalization()
        self.conv2 = Conv2D(filters=channels_out, kernel_size=ks, padding='same', kernel_initializer='he_normal')
        self.norm2 = BatchNormalization()
        self.actv = ReLU()
        
    def call(self, inp):
        
        x = self.actv(self.norm1(self.conv1(inp)))
        out = x = self.actv(self.norm2(self.conv2(x)))

        return out    
  
class EncoderLayer(tf.keras.layers.Layer):
    ''' Encoder Layer (down-sampling): 
        
        input: 
        - a tensor of shape (batch_size, high, width, channels)
        outputs: 
        - a tensor of shape (batch_size, high//2, width//2, channels*2)
        - a tensor to concatenate to the respective Decoder Layer of shape (batch_size, high, width, channels)
    '''
    def __init__(self, channels_in, channels_out, ks):
        super(EncoderLayer, self).__init__(name='EncoderLayer')
        
        self.channels_in = channels_in
        self.channels_out = channels_out
        
        self.double_conv = DoubleConvs(channels_in, ks)
        self.down = Conv2D(filters=channels_out, kernel_size=ks, strides=2, padding='same', kernel_initializer='he_normal')
        self.norm = BatchNormalization()
        self.actv = ReLU()
        
    def call(self, inp):
        
        assert self.channels_in*2 == self.channels_out
        
        out1 = self.double_conv(inp)
        out2 = self.actv(self.norm(self.down(out1)))
        
        return out1, out2
    
class DecoderLayer(tf.keras.layers.Layer):
    ''' Decoder Layer (up-sampling): 
        
        inputs: 
        - a tensor of shape (batch_size, high, width, channels)
        - a tensor to concatenate from the respective Encoder Layer.
        
        output: 
        - a tensor of shape (batch_size, high*2, width*2, channels//2)
    '''
    def __init__(self, channels_in, channels_out, ks):
        super(DecoderLayer, self).__init__(name='DecoderLayer')
        
        self.channels_in = channels_in
        self.channels_out = channels_out
        
        self.up = Conv2DTranspose(filters=channels_out, kernel_size=ks, strides=2, padding='same', kernel_initializer='he_normal')
        self.norm = BatchNormalization()
        self.actv = ReLU()
        self.double_conv = DoubleConvs(channels_out, ks)
        
    def call(self, inp, lay):
        
        assert self.channels_out == self.channels_in//2
        
        x = self.actv(self.norm(self.up(inp)))
        x = tf.concat([lay, x], axis=-1)
        out = self.double_conv(x)
              
        return out
    
class UNet(tf.keras.Model):
    ''' The UNet model.
    '''
    def __init__(self, n_classes, filters_start=64, ks=4, depth=3):
        super(UNet, self).__init__(name='UNet')
        
        self.depth = depth
        
        self.encoder = [EncoderLayer(filters_start*(2**i), filters_start*(2**(i+1)), ks) for i in range(depth)]
        self.bridge = DoubleConvs(filters_start*(2**depth), ks)
        self.decoder = [DecoderLayer(filters_start*(2**i), filters_start*(2**(i-1)), ks) for i in range(depth, 0, -1)]
        self.classifier = Conv2D(filters=n_classes, kernel_size=ks, padding='same', kernel_initializer='he_normal', activation='softmax')
        
    def call(self, inp):
        
        lays = []
        x = inp
        
        # Encoding
        for encoder_layer in self.encoder:
            lay, x = encoder_layer(x)
            lays.append(lay)
        
        # Bottleneck
        x = self.bridge(x)
                
        # Decoding  
        for decoder_layer in self.decoder:
            x = decoder_layer(x, lays.pop())
        
        # Classifing
        out = self.classifier(x)
            
        return out
