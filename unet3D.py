from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, Activation, BatchNormalization, Conv3DTranspose, Dropout, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

class unet(object):
    
    def __init__(self, img_size, Nclasses, class_weights, weights_name='myWeights.h5', Nfilter_start=64, depth=3):
        self.img_size = img_size
        self.Nclasses = Nclasses
        self.class_weights = class_weights
        self.weights_name = weights_name
        self.Nfilter_start = Nfilter_start
        self.depth = depth

        inputs = Input(img_size)
        
        def dice(y_true, y_pred, w=self.class_weights):
            y_true = tf.convert_to_tensor(y_true, 'float32')
            y_pred = tf.convert_to_tensor(y_pred, 'float32')

            num = 2 * tf.reduce_sum(tf.reduce_sum(y_true*y_pred, axis=[0,1,2,3])*w)
            den = tf.reduce_sum(tf.reduce_sum(y_true+y_pred, axis=[0,1,2,3])*w) + 1e-5

            return num/den
    
        def diceLoss(y_true, y_pred):
            return 1-dice(y_true, y_pred)          
        
        # This is a help function that performs 2 convolutions, each followed by batch normalization
        # and ReLu activations, Nf is the number of filters, filter size (3 x 3 x 3)
        def convs(layer, Nf):
            x = Conv3D(filters=Nf, kernel_size=3, kernel_initializer='he_normal', padding='same')(layer)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv3D(filters=Nf, kernel_size=3, kernel_initializer='he_normal', padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            return x
            
        # This is a help function that defines what happens in each layer of the encoder (downstream),
        # which calls "convs" and then down-sampling (2 x 2 x 2).
        def encoder_step(layer, Nf):
            y = convs(layer, Nf)
            x = Conv3D(filters=Nf, kernel_size=3, kernel_initializer='he_normal', padding='same', strides=2)(y)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            return y, x
            
        # This is a help function that defines what happens in each layer of the decoder (upstream),
        # which contains upsampling (2 x 2 x 2), 3D convolution (2 x 2 x 2), batch normalization, concatenation with 
        # corresponding layer (y) from encoder, and lastly "convs"
        def decoder_step(layer, layer_to_concatenate, Nf):
            x = Conv3DTranspose(filters=Nf, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal')(layer)
            x = BatchNormalization()(x)
            x = concatenate([x, layer_to_concatenate])
            x = convs(x, Nf)
            return x
            
        layers_to_concatenate = []
        x = inputs
        
        # Make encoder
        for d in range(self.depth-1):
            y,x = encoder_step(x, self.Nfilter_start*np.power(2,d))
            layers_to_concatenate.append(y)
            
        # Make bridge
        x = Dropout(0.2)(x)
        x = convs(x,self.Nfilter_start*np.power(2,self.depth-1))
        x = Dropout(0.2)(x)        
        
        # Make decoder
        for d in range(self.depth-2, -1, -1):
            y = layers_to_concatenate.pop()
            x = decoder_step(x, y, self.Nfilter_start*np.power(2,d))            
        
        # Make classificator
        final = Conv3D(filters=self.Nclasses, kernel_size=1, activation = 'softmax', padding='same', kernel_initializer='he_normal')(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=final)
        self.model.compile(loss=diceLoss, optimizer=Adam(lr=1e-4), metrics=['accuracy',dice])
        
    def train(self, train_gen, valid_gen, nEpochs):
        print('Training process:')       
        callbacks = [ModelCheckpoint(self.weights_name, save_best_only=True, save_weights_only=True),
                     EarlyStopping(patience=10)]
        
        history = self.model.fit(train_gen, validation_data=valid_gen, epochs=nEpochs, callbacks=callbacks)

        return history    
    
    def evaluate(self, test_gen):
        print('Evaluation process:')
        score, acc, dice = self.model.evaluate(test_gen)
        print('Accuracy: {:.4f}'.format(acc*100))
        print('Dice: {:.4f}'.format(dice*100))
        return acc, dice
    
    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred
    
    def calculate_metrics(self, y_true_flat, y_pred_flat):
        ''' This function calculates the metrics accuracy and Dice between two binary arrays.
        '''
        cm = tf.math.confusion_matrix(y_true_flat, y_pred_flat, num_classes=2).numpy()
        acc = np.trace(cm)/np.sum(cm)
        if cm[0,0] == len(y_true_flat):
            dice = np.nan
        else:
            dice = 2*cm[1,1]/(2*cm[1,1]+cm[1,0]+cm[0,1])
        
        return acc, dice
    
    def get_metrics(self, generator):
        ''' This function calculates the metrics accuracy and Dice for each image contained in the input generator.
        '''
        Nim = len(generator)*generator.batch_size
        ACC = np.empty((Nim, self.Nclasses))
        DICE = np.empty((Nim, self.Nclasses))
        n = 0
        for i in range(len(generator)):
            X_batch, y_batch = generator[i]
            y_pred = self.model.predict(X_batch)
            y_pred = to_categorical(tf.argmax(y_pred, axis=-1), self.Nclasses)
            
            RowsColumnsSlices = y_batch.shape[1]*y_batch.shape[2]*y_batch.shape[3]
            
            for b in range(X_batch.shape[0]):
                for c in range(Nclasses):
                    y_true_flat = tf.reshape(y_batch[b,:,:,c], (RowsColumnsSlices,))
                    y_pred_flat = tf.reshape(y_pred[b,:,:,c], (RowsColumnsSlices,))            

                    acc, dice = self.calculate_metrics(y_true_flat, y_pred_flat)
                    ACC[n,c] = acc
                    DICE[n,c] = dice

                n+=1

        return ACC, DICE
