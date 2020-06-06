import hashlib


def get_md5(filename):
    hash_obj = hashlib.md5()
    with open(filename, 'rb') as f:
        hash_obj.update(f.read())
    return hash_obj.hexdigest()
