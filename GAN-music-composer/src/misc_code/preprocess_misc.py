
'''
Misc code:
    def get_spectrograms_and_label_id_split(self, waveform, label):
        label_id = tf.argmax(label == MUSIC_DIRS)
        spectrograms = tf.Tensor()
        for waveform in waveforms:
            zero_padding = tf.zeros([int(AUDIO_LEN)] - tf.shape(waveform), dtype=tf.float32)
            waveform = tf.cast(waveform, tf.float32)
            equal_length = tf.concat([waveform, zero_padding], 0)
            spectrogram = tf.signal.stft(
                            equal_length, frame_length=2048, frame_step=1024)
            spectrograms.append(tf.abs(spectrogram))
        return tf.convert_to_tensor(spectrograms), label_id
    def decode_mp3_split(self, audio_binary):
        audios = []
        full_audio = tfio.audio.decode_mp3(audio_binary)[:,0]
        full_audio_len =  full_audio.shape[0]
        num_audio_parts = full_audio_len//AUDIO_LEN
        for i in range(num_audio_parts):
            start = i*AUDIO_LEN
            end = start + AUDIO_LEN
            this_audio = full_audio[start:end]
            audios.append(this_audio)
        return tf.convert_to_tensor(audios)
    '''
