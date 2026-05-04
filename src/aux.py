def displayAsSet(lst):
    lst = [str(x) for x in lst]
    return '{' + ', '.join(lst) + '}'

def displayOrdered(set):
    return '{' + ', '.join(str(x) for x in sorted(set)) + '}'