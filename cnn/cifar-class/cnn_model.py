import tensorflow as tf
from tensorflow.keras import layers,models

class CNNEverything:
    def __init__(self,inp_shape, filt_sizes=[],
                                kernel_sizes=[],\
                                act ='relu',\
                                use_batchnorm = None,\
                                use_dropout = None
                                ):
        if not len(filt_sizes)==len(kernel_sizes):
            raise ValueError('Bad shapes: num filter sizes and kernel sizes must be same. You entered {} filters,  {} kernels'.format(len(filt_sizes),len(kernel_sizes)))
        if not use_batchnorm:
            use_batchnorm = [False]*len(filt_sizes)
        if not use_dropout:
            use_dropout = [False]*len(filt_sizes)
        self.inp_shape = inp_shape #tuple - (x_size, y_size, channels) of images
        self.filt_sizes = filt_sizes #list of ints, list len = num hidden blocks
        self.kernel_sizes = kernel_sizes #list of tuple of ints, list len = num hidden blocks,tuple eg:(3,3)
        self.act = act  # activation to use (default_relu)
        self.use_batchnorm = use_batchnorm #by default, conv2D and max pool 2D will be added to every hidden block; use these to add other layers
        self.use_dropout = use_dropout

    def create(self):
        self.model = models.Sequential()
        for i in range(len(self.filt_sizes)):
            if i==0:
                self.model.add(layers.Conv2D(self.filt_sizes[i], self.kernel_sizes[i], activation='relu', input_shape=self.inp_shape))
            else:
                self.model.add(layers.Conv2D(self.filt_sizes[i], self.kernel_sizes[i], activation='relu'))
            self.model.add(layers.MaxPooling2D((2, 2)))
            if self.use_batchnorm[i]:
                self.model.add(layers.BatchNormalization())
            if self.use_dropout[i]:
                self.model.add(layers.Dropout(rate=DROPOUT_RATE))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(10))
        return self.model 


    def compile(self, optimizer='adam',\
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),\
                            metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, 
                        loss=loss,\
                        metrics=metrics)
