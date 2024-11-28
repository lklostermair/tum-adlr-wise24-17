import h5py

def load_tactmat_data(path):
    with h5py.File(path, 'r') as dataset:
        samples = dataset['samples'][:]
        materials = dataset['materials'][:]
        materials = [m.decode('utf-8') for m in materials]
    train_samples = samples[:, :80, ...]
    test_samples = samples[:, 80:, ...]
    return train_samples, test_samples, materials
