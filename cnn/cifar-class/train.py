

from read_ds import DatasetEverything
from cnn_model import CNNEverything
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
import os
import sys

CIFAR10_CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
DROPOUT_RATE = 0.5

def parse_my_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_name',help='Name of the dataset to read',default='cifar10')
    parser.add_argument('--ds_source',help='Where to read this Dataset from? tf.keras.datasets/custom',default='tf_keras_datasets')
    parser.add_argument('--model_save_path',help='Path to save the trained models',default='.')
    parser.add_argument('--train',help='Whether to train model from scratch or use pre-trained',action='store_true')
    parser.add_argument('--epochs',help='Num epochs to train the model', default=10, type=int)
    parser.add_argument('--trained_model',help='Path to load the trained model from')
    args = parser.parse_args()
    return args


class TrainEverything:
    def __init__(self,  model=None,\
                        train_images=None, 
                        train_labels=None,\
                        test_images=None,\
                        test_labels=None,\
                        epochs=10,
                        save_path='.'):
        if not (model or  train_images or train_labels or test_images or test_labels):
            raise ValueError('Pass a valid model, train and test datasets to perform training')
        self.model = model
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels
        self.epochs = epochs
        self.save_path=save_path
    
    def callbacks(self, monitor='val_loss',
                        save_weights_only=True,
                        save_best_only=True):
        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath = os.path.join(self.save_path,'model_{epoch:02d}-{val_loss:.2f}.hdf5'),
            save_weights_only = save_weights_only,
            verbose = 0,
            monitor = monitor,
            save_best_only = save_best_only)
        self.callbacks = [self.model_checkpoint_callback]   
 
    def train(self):
        self.history = self.model.fit(self.train_images, self.train_labels, epochs=self.epochs, 
                    validation_data=(self.test_images, self.test_labels),
                    callbacks=self.callbacks)
        return self.history 

    
    def plot_train_hist(self):
        plt.plot(self.history.history['accuracy'], label='accuracy')
        plt.plot(self.history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.show()


class EvalEverything:
    def __init__(self, model,
                    test_images,test_labels):
        self.model = model
        self.test_images = test_images
        self.test_labels = test_labels


    def evaluate(self):        
        self.test_loss, self.test_acc = self.model.evaluate(self.test_images,  self.test_labels, verbose=2)       

def main():
    args = parse_my_args()
    ds = DatasetEverything(args.ds_name, args.ds_source)
    (train_images, train_labels), (test_images, test_labels) = ds.read_dataset() 
    if args.train:
        train_images, test_images = ds.normalize_imgs() # normalize pixel values to be between 0 and 1
        ds.sneak_peek(CIFAR10_CLASS_NAMES)
        cnn = CNNEverything((32,32,3),filt_sizes=[32,64,64],kernel_sizes=[(3,3),(3,3),(3,3)])
        model = cnn.create()
        cnn.compile()
        model.summary()    
        trainer = TrainEverything(model, train_images, train_labels, test_images, test_labels, 
                                save_path=args.model_save_path, epochs=args.epochs)
        trainer.callbacks(monitor='val_loss',save_weights_only=False,save_best_only=True)
        history = trainer.train()   
        trainer.plot_train_hist()
    elif args.trained_model:
        model = tf.keras.models.load_model(args.trained_model)
    else:
        raise ValueError('Either make train=True to train a model from scratch or enter a valid model path in trained_model')
        
    evaluate = EvalEverything( model, test_images, test_labels)
    evaluate.evaluate()    

if __name__=="__main__":
    main()
