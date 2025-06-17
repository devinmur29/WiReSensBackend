from utils import tactile_reading
pressure, fc, ts = tactile_reading("./recordings/esp1.hdf5")
print(fc/(ts[-1]-ts[0]))