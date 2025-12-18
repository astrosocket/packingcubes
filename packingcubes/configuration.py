from array import array


def determine_field_format():
    formats = ["H", "I", "L", "Q"]
    for f in formats:
        if array(f).itemsize == 4:
            return f
    if array(formats[0]).itemsize > 4:
        return formats[0]
    raise NotImplementedError("None of the included array formats are long enough!")


FIELD_FORMAT = determine_field_format()
