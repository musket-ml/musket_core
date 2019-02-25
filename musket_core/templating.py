import ast
import operator as op

# supported operators
operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
             ast.Div: op.truediv, ast.Pow: op.pow, ast.BitXor: op.xor,
             ast.USub: op.neg}

def eval_expr(expr,globals):
    """
    >>> eval_expr('2^6')
    4
    >>> eval_expr('2**6')
    64
    >>> eval_expr('1 + 2*3**(4^5) / (6 + -7)')
    -5.0
    """
    return eval_(ast.parse(expr, mode='eval').body,globals)

def eval_(node,globals):
    if isinstance(node, ast.Num): # <number>
        return node.n
    elif isinstance(node, ast.BinOp): # <left> <operator> <right>
        return operators[type(node.op)](eval_(node.left,globals), eval_(node.right,globals))
    elif isinstance(node, ast.UnaryOp): # <operator> <operand> e.g., -1
        return operators[type(node.op)](eval_(node.operand,globals))
    elif isinstance(node,ast.Name):
        return globals[node.id]
    else:
        raise TypeError(node.id)


def resolveTemplates(data,params):
    if isinstance(data, dict):
        return { x:resolveTemplates(data[x],params) for x in data}
    if isinstance(data,list):
        return [resolveTemplates(x,params) for x in data]
    for param in params:
        if isinstance(data,str):
            if param in data:
                try:
                    r=eval_expr(data,params)
                    return r
                except:
                    pass
    return data

#print(eval_expr("3*hello",{"hello":5}))