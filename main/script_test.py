import pickle

if __name__ == '__main__':
    token2index = pickle.load(open('../data/meta_data/residue2idx.pkl', 'rb'))
    print('token2index', token2index)
    print('Script Test Passed')
