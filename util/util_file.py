def read_tsv_data(filename, skip_first=True):
    sequences = []
    labels = []
    with open(filename, 'r') as file:
        if skip_first:
            next(file)
        for line in file:
            if line[-1] == '\n':
                line = line[:-1]
            list = line.split('\t')
            sequences.append(list[2])
            labels.append(int(list[1]))
    return [sequences, labels]
