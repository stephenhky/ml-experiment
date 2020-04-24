

def adding_no_features(datum):
    # add values here
    return datum

def convert_label_to_str(datum, label):
    if isinstance(datum[label], str):
        pass
    datum[label] = str(datum[label])
    return datum

