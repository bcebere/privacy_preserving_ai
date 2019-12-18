import phe
import torch
import numpy as np


class KeyMaster():
    def __init__(self):
        self.public, self.private = phe.generate_paillier_keypair(n_length=256)

    def public_key(self):
        return self.public

    def encrypt(self, plain):
        return self.public.encrypt(float(plain))

    def decrypt(self, enc):
        return self.private.decrypt(enc)

    def encrypt_tensor_to_numpy(self, tensor):
        arr = tensor.cpu().detach().numpy()
        vfunc = np.vectorize(self.encrypt)
        return vfunc(arr)

    def decrypt_nparray(self, enc):
        vfunc = np.vectorize(self.decrypt)
        return vfunc(enc)
