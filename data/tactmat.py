import h5py

def load(filename):
    with  h5py.File(filename, 'r') as dataset:
        samples = dataset['samples'][:]
        materials = dataset['materials'][:]
        materials = [m.decode('utf-8') for m in materials]
    return (samples, materials)


