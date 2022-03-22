
def flatten(s: list) -> list:
    if s == []:
        return s

    if isinstance(s[0], list):
        return flatten(s[0]) + flatten(s[1:])
        
    return s[:1] + flatten(s[1:])