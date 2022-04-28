
from clint.textui import progress
import requests

def get_data_from_url(dataset:str):

    horse2zebra_url = 'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip'
    res = requests.get(horse2zebra_url,stream=True)
    total_length = int(res.headers.get('content-length'))
    # print("123")
    with open("hz.zip", "wb") as pypkg:
        for chunk in progress.bar(res.iter_content(chunk_size=1024*1024), expected_size=(total_length/(1024*1024) ) + 1, width=100):
            if chunk:
                # print("write i")
                pypkg.write(chunk)
get_data_from_url('horse2zebra')
