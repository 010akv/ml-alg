# GAN Composer - WIP
- train a GAN to generate songs in the Tamil language

## Workflow
- Crawl through movies and web scrape Tamil language mp3 songs from [friendstamilmp3](https://friendstamilmp3.in)
- Preprocess the audio files
    - decode 
    - make a spectrogram using the STFT algorithm 
        - spectrogram 
            - visual representation (image) of frequencies vs. time along with amplitude info
            - colors represent the thrid dimension (amplitude)
            - so, how do you read a spectrogram? Like so - "at 7am, frequencies between 4-5Hz were observed at a large amplitude"
        - STFT - Short Time Fourier Transform
            - sequence of Fourier Transforms over a windowed signal
            - shows how the frequency of a signal changes over time
            - Fourier Transform provides frequency information averaged over the entire signal duration
- make TFRecords with spectrogram images
- train a GAN 
