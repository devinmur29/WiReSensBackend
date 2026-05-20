from wiresens_backend.utils import tactile_reading
pressure, fc, ts = tactile_reading("./test.hdf5")
print(fc/(ts[-1]-ts[0]))