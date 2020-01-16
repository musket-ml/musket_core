'''
@author: 32kda
'''

after_download_callbacks=[]

def after_download(method):
    after_download_callbacks.append(method)
    return method
    pass

def get_after_download_callbacks():
    return after_download_callbacks