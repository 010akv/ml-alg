

from read_ds import DatasetEverything
from tensorflow.keras import layers,models
import argparse

CIFAR10_CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
DROPOUT_RATE = 0.5

def parse_my_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_name',help='Name of the dataset to read')
    parser.add_argument('--ds_source',help='Where to read this Dataset from? tf.keras.datasets/custom')
    args = parser.parse_args()
    return args

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

    def make_model(self):
        model = models.Sequential()
        for i in range(len(self.filt_sizes)):
            if i==0:
                model.add(layers.Conv2D(self.filt_sizes[i], self.kernel_sizes[i], activation='relu', input_shape=self.inp_shape))
            else:
                model.add(layers.Conv2D(self.filt_sizes[i], self.kernel_sizes[i], activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))
            if self.use_batchnorm[i]:
                model.add(layers.BatchNormalization())
            if self.use_dropout[i]:
                model.add(layers.Dropout(rate=DROPOUT_RATE))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10))
        return model 
        
def main():
    args = parse_my_args()
    ds = DatasetEverything(args.ds_name, args.ds_source)
    (train_images, train_labels), (test_images, test_labels) = ds.read_dataset() 
    train_images, test_images = ds.normalize_imgs() # normalize pixel values to be between 0 and 1
    print(train_labels.shape)
    ds.sneak_peek(CIFAR10_CLASS_NAMES)
    cnn = CNNEverything((32,32,3),filt_sizes=[32,64,64],kernel_sizes=[(3,3),(3,3),(3,3)])
    model = cnn.make_model()
    model.summary()    


if __name__=="__main__":
    main()
