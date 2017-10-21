from datetime import datetime

def sieno_converter (input):
    value = str(input).lower()
    if value == "si":
        return 1
    elif value == "no":
        return 0
    else:
        raise ValueError("%s is not in a valid format" % input)


def date_converter (input):
    formats = ["%d/%m/%Y", "%Y-%m-%d 00:00:00"]
    for format in formats:
        try:
            return datetime.strptime(input, format).date()
        except ValueError:
            pass

    raise ValueError("%s is not in a valid format" % input)

def decimal_converter (input) :
    pass
