import pickle

def load_pkl(file_path):
    """Load pickle file.
    Args:
        file_path (str): File path.
    Returns:
        obj: Loaded object.
    """
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj


if __name__ == '__main__':
    file_path = 'extra_data/val/42.pkl'
    obj = load_pkl(file_path)
    print(obj)