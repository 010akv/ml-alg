from urllib.request import Request, urlopen
import os
from bs4 import BeautifulSoup
import requests

ARR_URL_PREFIX = 'https://friendstamilmp3.in/index.php?page=A%20R%20Rahman%20Hits&spage='
MAX_ALLOWED_SONGS = 100000

class DownloadMp3:
    def __init__(self, base_url, download_dir, prefix):
        self.base_url       = base_url
        self.download_dir   = download_dir
        self.prefix         = prefix
        if not os.path.exists(self.download_dir):
            os.mkdir(self.download_dir)        
        if not os.path.exists(os.path.join(self.download_dir, 'songs')):
            os.mkdir(os.path.join(self.download_dir, 'songs'))
    
    def _get_movie_names(self):
        req             = Request(self.base_url)
        html_page       = urlopen(req)
        soup            = BeautifulSoup(html_page, "lxml")
        movie_names     = []
        for link in soup.findAll('a'):
            this_link = link.get('href')
            if '&spage=' in this_link:
              movie_names.append(this_link.split('&spage=')[-1])
        print('Getting movie names...')
        print('Got {} total movies, sneak peek: {}....'.format(len(movie_names),movie_names[:5]))
        return movie_names

    def get_all_urls(self, \
                        num_songs = 10,\
                        filepath='arr_songs_urls.csv'):
        
        print('Downloading {} urls'.format(num_songs))
        
        if not isinstance(num_songs,int) or num_songs>MAX_ALLOWED_SONGS:
            raise ValueError('num_songs must be a valid integer between 0 and 100. You entered {} - invalid')
        
         
        song_idx        = 0
        all_done        = 0
        done_movies     = []
        movie_names     = self._get_movie_names()
        all_urls        = []
        for movie in movie_names:
            if all_done==1:
                if filepath:
                    with open(filepath,'w') as f:
                        f.write('\n'.join(all_urls))
                return all_urls
 
            link        = self.prefix + movie.replace(' ','%20')
            req         = Request(link)
            html_page   = urlopen(req)
            soup        = BeautifulSoup(html_page, "lxml")
        
            for link in soup.findAll('a'):
                
                if song_idx >= num_songs:
                    all_done = 1
                    break
                
                this_link = link.get('href')

                if 'http' in this_link and this_link.endswith('.mp3'):
                    this_link           = this_link.replace(' ','%20')
                    this_link_parts     = this_link.split('/')
                    all_urls.append(this_link)
                    song_idx += 1
                
        if filepath:
            with open(filepath,'w') as f:
                f.write('\n'.join(all_urls))
    
        return all_urls

    def download_url(self, url):
        url_parts           = url.split('/')
        song_filename       = os.path.join(self.download_dir, 'songs', str(url_parts[-1].replace('%20','_')))
        if not os.path.exists(song_filename):
            with requests.Session() as req:
                download        = req.get(url)
                if download.status_code == 200:
                    with open(song_filename, 'wb') as f:
                        f.write(download.content)
        


