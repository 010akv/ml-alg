import time
import argparse 
import os
from threading import Thread
from download_mp3_utils import DownloadMp3

MAX_ALLOWED_SONGS = 1000
BASE_URL = 'https://friendstamilmp3.in/index.php?page=A%20R%20Rahman%20Hits'
ARR_URL_PREFIX='https://friendstamilmp3.in/index.php?page=A%20R%20Rahman%20Hits&spage='
URL_FILENAME = 'arr_songs_urls.csv'
THREAD_BATCH_SIZE = 100

def parse_my_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download_urls',type=int,default=0, help='Make this 1 to download all URLs into a file')
    parser.add_argument('--download_mp3s',type=int,default=0, help='Make this 1 to dpwnload MP3 songs')
    parser.add_argument('--multithreaded',type=int,default=0, help='Make this 1 to make downloads multhithreaded and faster')
    parser.add_argument('--num_songs',type=int,default=1,help='Num songs to download, must be an integer 0 - 1000')
    parser.add_argument('--download_dir',type=str,default='/Users/anu/datasets/tamil_mp3/arr', help='Where to download the URLs and mp3s to')
    args = parser.parse_args()
    return args


def validate(args):
    download_urls = args.download_urls
    download_mp3s = args.download_mp3s
    multithreaded = args.multithreaded
    num_songs = args.num_songs

    if not (download_urls == 0 or download_urls == 1):
        raise ValueError('--download_urls must be either 0 or 1')

    if not (download_mp3s == 0 or download_mp3s == 1):
        raise ValueError('--download_mp3s must be either 0 or 1')

    if not (multithreaded == 0 or multithreaded == 1):
        raise ValueError('--multithreaded must be either 0 or 1')

    if num_songs < 0 or num_songs > MAX_ALLOWED_SONGS:
        raise ValueError('--num_songs must be an integer between 0 and {}'.format(MAX_ALLOWED_SONGS))
    
    if not isinstance(num_songs,int):
        raise TypeError('--num_songs must be an integer')
    


def download_urls(num_songs, download_dir, obj_download_mp3):
    return obj_download_mp3.get_all_urls(num_songs, filepath = os.path.join(download_dir, URL_FILENAME))

def download_mp3s(urls, obj_download_mp3):
    _ = [obj_download_mp3.download_url(url) for url in urls]

def download_mp3_batch(batch_urls, obj_download_mp3):
    batch_threads = []
    for url in batch_urls:
        this_thread = Thread(target=obj_download_mp3.download_url, args = (url, ))
        batch_threads.append(this_thread)
        this_thread.start()

    for thread in batch_threads:
        thread.join()
    
def download_mp3s_multithreaded(urls, obj_download_mp3):
    num_batches = len(urls) // THREAD_BATCH_SIZE
    for batch in range(num_batches):
        start = batch * THREAD_BATCH_SIZE
        end = start + THREAD_BATCH_SIZE
        print('Downloading batch {} - {} to {}'.format(batch, start, end))
        download_mp3_batch(urls[start:end], obj_download_mp3)

    start = num_batches*THREAD_BATCH_SIZE
    end = start + len(urls) % THREAD_BATCH_SIZE
    download_mp3_batch(urls[start:end], obj_download_mp3)
     

def main():
    start = time.time()
    args = parse_my_args()
    validate(args)    
    print('Got args in {} seconds'.format(time.time()-start))
    obj_download_mp3 = DownloadMp3(BASE_URL, args.download_dir, ARR_URL_PREFIX)
    if args.download_urls:
        urls = download_urls(args.num_songs, args.download_dir, obj_download_mp3)
        print('Gor URLs in {} seconds'.format(time.time()-start)) 
    if args.download_mp3s:
        if not args.multithreaded:
            download_mp3s(urls, obj_download_mp3)   
            print("Downloaded mp3s in {} seconds".format(time.time()-start))
        else:
            download_mp3s_multithreaded(urls,  obj_download_mp3)   
            print('Downloaded mp3s in {} seconds'.format(time.time()-start))
    print('Total time {} seconds'.format(time.time()-start))

if __name__=="__main__":
    main()
