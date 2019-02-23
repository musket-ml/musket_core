

def resolveTemplates(data,params):
    if isinstance(data, dict):
        return { x:resolveTemplates(data[x],params) for x in data}
    if isinstance(data,list):
        return [resolveTemplates(x,params) for x in data]
    if data in params:
        return params[data]
    return data

