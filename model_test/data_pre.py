'''
对比两个多肽数据库的序列重复情况:
startPepDB和PEPBIO-UWM
'''
import numpy as np
import pandas as pd


def load_fasta(filename, skip_first=False):
    with open(filename, 'r') as file:
        if skip_first:
            next(file)
        content = file.read()

    content_split = content.split('>')

    data = []
    for record in content_split:
        # print('-' * 50)
        # print('record:', record)
        record_split = record.split('\n')
        # print('record_split:', record_split)
        seq_seg_list = [record_split[i] for i in range(len(record_split)) if 0 < i < len(record_split) - 1]
        seq = ''.join(str(i) for i in seq_seg_list)
        # print('seq:', seq)
        if seq != '':
            data.append(seq)
    return data

def write_tsv_format_data(tsv_filename, labels, sequences):
    if len(labels) == len(sequences):
        with open(tsv_filename, 'w') as file:
            file.write('index\tlabel\ttext\n')
            for i in range(len(labels)):
                file.write('{}\t{}\t{}\n'.format(i, labels[i], sequences[i]))
        return True
    return False

if __name__ == '__main__':
    filename = 'Train_negative.txt'
    Train_negative = load_fasta(filename)
    print('Train_negative', len(Train_negative), len(np.unique(Train_negative)), Train_negative)

    filename = 'Train_positive.txt'
    Train_positive = load_fasta(filename)
    print('Train_positive', len(Train_positive), len(np.unique(Train_positive)), Train_positive)

    filename = 'Validate_negative.txt'
    Validate_negative = load_fasta(filename)
    print('Validate_negative', len(Validate_negative), len(np.unique(Validate_negative)), Validate_negative)

    filename = 'Validate_positive.txt'
    Validate_positive = load_fasta(filename)
    print('Validate_positive', len(Validate_positive), len(np.unique(Validate_positive)), Validate_positive)

    write_tsv_format_data('Train_negative.tsv', [0]*len(Train_negative), Train_negative)
    write_tsv_format_data('Train_positive.tsv', [1] * len(Train_positive), Train_positive)
    write_tsv_format_data('Validate_negative.tsv', [0] * len(Validate_negative), Validate_negative)
    write_tsv_format_data('Validate_positive.tsv', [1] * len(Validate_positive), Validate_positive)