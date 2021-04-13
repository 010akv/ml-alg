import os
from preprocess import *
import argparse
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution



print(tf.executing_eagerly())

def parse_my_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mp3_dir',default='/root/datasets/arr/',help='Path to mp3 files')
    args = parser.parse_args()
    return args

def validate(args):
    for root,_,filenames in os.walk(args.mp3_dir):
        for filename in filenames:
            if filename.endswith('.mp3'):
                return 
    raise ValueError('The directory must have at least one mp3 file')    



def decode_mp3_local(audio_binary):
    return tfio.audio.decode_mp3(audio_binary)[:,0]

def get_waveforms_local(filepath):
    audio_binary = tf.io.read_file(filepath)
    waveforms = decode_mp3_local(audio_binary)
    return waveforms

def test_map_fn(filepath):
    return tf.io.read_file(filepath)

def main():
    args = parse_my_args()
    validate(args)
    train, test, val = split(args.mp3_dir)
    files_ds = tf.data.Dataset.from_tensor_slices(train)
    for this in files_ds.take(1):
        print(this)
        break
    waveforms_ds = files_ds.map(get_waveforms_local)
    for wave in waveforms_ds.take(1):
        print(wave)
        break


if __name__=='__main__':
    main()
