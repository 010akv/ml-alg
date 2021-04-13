import tensorflow as tf
import tensorflow_io as tfio
import os
from tensorflow.python.framework.ops import disable_eager_execution
import argparse

print(tf.executing_eagerly())
AUDIO_LEN = 20000000
MUSIC_DIRS = ['arr']
SHARD_SIZE= 2

def parse_my_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mp3_dir',default='/root/dataset/arr/songs',help='Path to mp3 songs')
    parser.add_argument('--tfr_path', default='/root/dataset/arr/tfrecords/arr.tfrecord',help='Path to write the TFRecords')
    args = parser.parse_args()
    return args    

def validate_args(args):
    if not os.path.exists('/'.join(args.tfr_path.split('/')[:-1])):
        raise FileNotFoundError('Enter a valid path to write TFRecords. {} not found'.format('/'.join(args.tfr_path.split('/')[:-1])))
    
    if not os.path.exists(args.mp3_dir):
        raise FileNotFoundError('Enter a valid mp3_dir. {} not found'.format(args.mp3_dir))
    
    for root,_,files in os.walk(args.mp3_dir):
        for filename in files:
            if filename.endswith('.mp3'):
                return
    raise ValueError('No valid mo3 files found in mp3_dir - {}'.format(args.mp3_dir))

class PreprocessMp3:
    def __init__(self):
        pass
    
    def _decode_mp3(self, audio_binary):
        return tfio.audio.decode_mp3(audio_binary)[:,0]

    
    def _get_label(self, filepath):
        parts = tf.strings.split(filepath, os.path.sep)
        return parts[-1] 
    
    
    def _get_waveform(self, filepath):
        audio_binary = tf.io.read_file(filepath)
        waveform = self._decode_mp3(audio_binary)
        label = self._get_label(filepath)
        return waveform, label

    def _get_spectrogram_and_label_id(self, waveform, label):
        label_id = tf.argmax(label == MUSIC_DIRS)
        zero_padding = tf.zeros([int(AUDIO_LEN)] - tf.shape(waveform), dtype=tf.float32)
        waveform = tf.cast(waveform, tf.float32)
        equal_length = tf.concat([waveform, zero_padding], 0)
        spectrogram = tf.signal.stft(
                        equal_length, frame_length=8192, frame_step=4096)
        return spectrogram, label_id
    
    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def _int64_feature(self, value):
         return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    def _serialize_array(self, array):
        array = tf.io.serialize_tensor(array)
        return array
    
    def _to_tfexample(self, audio, label):
        feature = {
            'spectrogram': self._bytes_feature(audio),  
            'label': self._int64_feature(label)  
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))
    
    def _make_tfrecord(self, spec_ds, tfr_path):
        with tf.io.TFRecordWriter(tfr_path) as out:
            for spectrogram, label in spec_ds:
                serialized_array = self._serialize_array(spectrogram.numpy())
                tfexample = self._to_tfexample(serialized_array.numpy(), label.numpy())
                out.write(tfexample.SerializeToString())
    
    def dataset_pipeline(self, files):
        AUTOTUNE = tf.data.AUTOTUNE 
        files_ds = tf.data.Dataset.from_tensor_slices(files)
        waveforms_ds = files_ds.map(self._get_waveform, num_parallel_calls = AUTOTUNE)
        spectrograms_ds = waveforms_ds.map(self._get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)
        return spectrograms_ds
    
    def split(self, mp3_dir, train_perc = 60, val_perc = 20):
        filenames = tf.io.gfile.glob(str(mp3_dir) + '/*')
        filenames = tf.random.shuffle(filenames)
        num_samples = len(filenames)
        print('Number of total examples:', num_samples)
        
        num_tr = int(0.6*(num_samples))
        num_val = int(0.2*(num_samples))
        
        train_files = filenames[:num_tr]
        val_files = filenames[num_tr:num_tr+num_val]
        test_files = filenames[num_tr+num_val:]
        
        print('Training set size', len(train_files))
        print('Validation set size', len(val_files))
        print('Test set size', len(test_files)) 
        return train_files, test_files, val_files
            
    def make_tfrecords(self, files, tfr_path):
        num_shards = len(files) // SHARD_SIZE
        for shard in range(num_shards):
            start = shard * SHARD_SIZE
            end = start + SHARD_SIZE
            print('Shard {}, {} - {}'.format(shard, start, end))
            spec_ds = self.dataset_pipeline(files[start:end])
            this_path = tfr_path.split('.')[0] + '-%.5d-of-%.5d.' % (shard,  num_shards - 1 ) + tfr_path.split('.')[1]
            self._make_tfrecord(spec_ds, this_path)
    
    def _parse_tfr_element(self,element):
        feature = {
            'spectrogram': tf.io.FixedLenFeature([], tf.string), 
            'label': tf.io.FixedLenFeature([], tf.int64)
          }
        example_message = tf.io.parse_single_example(element, feature)
        spectrogram = example_message['spectrogram'] # get byte string
        spectrogram = tf.io.parse_tensor(spectrogram, out_type=tf.complex64) # restore 2D array from byte string
        label = example_message['label']
        return spectrogram, label

    def clean_up(self, filepaths, err_songs_file):
        if os.path.exists(err_songs_file):
            os.remove(err_songs_file)
        with open(err_songs_file, 'a') as f:
            for mp3_path in filepaths:
                try:
                    ave, label = self._get_waveform(mp3_path)
                except:
                    mp3_path = str(mp3_path.numpy().decode('utf-8')+'\n')
                    f.write(mp3_path)

        return 0



def main():
    args = parse_my_args()
    validate_args(args)
    obj = PreprocessMp3()
    train_files, test_files, val_files = obj.split(args.mp3_dir)
    err_songs_file='/root/dataset/arr/err_songs.txt'
    obj.clean_up(train_files, str(err_songs_file.split('.')[0] + '_train_.' + err_songs_file.split('.')[1]))
            #try:
            #    _ = self._get_waveform(mp3_path)
            #except:
            #    mp3_path = str(mp3_path.numpy().decode('utf-8'))
            #    f.write(mp3_path)
            #    f.write('\n')
    #AUTOTUNE = tf.data.AUTOTUNE 
    #files_ds = tf.data.Dataset.from_tensor_slices(train_files)
    #_ =    files_ds.map(obj.find_err_mp3s, num_parallel_calls=AUTOTUNE)
    #waveforms_ds = files_ds.map(self._get_waveform, num_parallel_calls = AUTOTUNE)
    #spectrograms_ds = waveforms_ds.map(self._get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)
    #spec_ds = obj.dataset_pipeline(train_files)
    #for spec, label in spec_ds:
    #    print(label)
    #obj.make_tfrecords(train_files, args.tfr_path)
   # tfr_ds = tf.data.TFRecordDataset('/root/dataset/arr/tfrecords/arr-00000-of-00000.tfrecord')
   # spec_ds = tfr_ds.map(obj._parse_tfr_element)
   # for spec, label in spec_ds:
   #     print(spec.numpy().shape, label.numpy())

if __name__=="__main__":
    main()

