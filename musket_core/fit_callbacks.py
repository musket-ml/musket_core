'''
@author: 32kda
'''

after_fit_callbacks=[]

def after_fit(method):
    after_fit_callbacks.append(method)
    return method
    pass

def get_after_fit_callbacks():
    return after_fit_callbacks