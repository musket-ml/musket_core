import  keras

def keras_metric(func):
    keras.utils.get_custom_objects()[func.__name__]=func
    return func



def final_metric(func):
    func.final_metric=True
    return func