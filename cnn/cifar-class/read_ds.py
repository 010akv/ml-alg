from tensorflow.keras import datasets
import matplotlib.pyplot as plt
import argparse

TF_KERAS_DATASETS = [ds for ds in dir(datasets) if not ds.startswith('_')]
CIFAR10_CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def parse_my_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_name',help='Name of the dataset to read')
    parser.add_argument('--ds_source',help='Where to read this Dataset from? tf.keras.datasets/custom')
    args = parser.parse_args()
    return args

class DatasetEverything:
    def __init__(self,ds_name, ds_source):
        if ds_source == 'tf_keras_datasets':
            if not ds_name in TF_KERAS_DATASETS:
                raise ValueError('Enter one of the tf keras datasets {} or change ds_source to custom'.format(TF_KERAS_DATASETS))           
        self.ds_name = ds_name
        self.ds_source = ds_source
    
    def read_dataset(self):
        if self.ds_source=='tf_keras_datasets':
            lcls = locals()
            read_command = 'out = datasets.{}.load_data()'.format(self.ds_name) # exec requires output to be stored in a variable
            exec(read_command, globals(), lcls)                                 # to make the variable 'out' available outside of exec, use locals(); out = lcls['out']
            (self.train_images, self.train_labels), (self.test_images, self.test_labels) = lcls['out']
            return (self.train_images, self.train_labels), (self.test_images, self.test_labels) 
    
    def normalize_imgs(self):
        self.train_images, self.test_images = self.train_images / 255.0, self.test_images / 255.0
        return self.train_images, self.test_images

    def sneak_peek(self, ds_class_names):
        print('Total num images: train - {}, test - {}'.format(self.train_images.shape[0], self.test_images.shape[0]))
        print('Let\'s look at one image:')
        print('Image ID:0','Resolution: ', self.train_images[0].shape)
        plt.figure(figsize=(10,10))
        for i in range(25):
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.train_images[i], cmap=plt.cm.binary)
            # The CIFAR labels happen to be arrays, 
            # which is why you need the extra index
            plt.xlabel(ds_class_names[self.train_labels[i][0]])
        plt.show()


class CNNEverything:
    def __init__(self,inp_shape, filt_sizes=[],
                                kernel_sizes=[],\
                                act ='relu',\
                                use_batchnorm = False,\
                                use_dropout = False
                                ):
        self.inp_shape = inp_shape #tuple - (x_size, y_size, channels) of images
        self.filt_sizes = filt_sizes #list of ints, list len = num hidden blocks
        self.kernel_sizes = kernel_sizes #list of tuple of ints, list len = num hidden blocks,tuple eg:(3,3)
        self.act = act  # activation to use (default_relu)
        self.use_batchnorm = use_batchnorm #by default, conv2D and max pool 2D will be added to every hidden block; use these to add other layers
        self.use_dropout = use_dropout


def main():
    args = parse_my_args()
    ds = DatasetEverything(args.ds_name, args.ds_source)
    (train_images, train_labels), (test_images, test_labels) = ds.read_dataset() 
    train_images, test_images = ds.normalize_imgs() # normalize pixel values to be between 0 and 1
    ds.sneak_peek(CIFAR10_CLASS_NAMES)    


if __name__=="__main__":
    main()



