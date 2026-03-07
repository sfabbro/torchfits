import time
import re

dummy_data = {
    1: "projp1 = 0.5 projp2 = -0.1 other = 1.0 " * 10,
    2: "projp3=1.2  PROJP4 = -3.4 " * 10,
}

# original
def extract_zpx_params_old(wat_data):
    params = {}
    pattern = re.compile(r"projp(\d+)\s*=\s*(\S+)", re.IGNORECASE)
    for axis, wstr in wat_data.items():
        matches = pattern.findall(wstr)
        for m in matches:
            idx = int(m[0])
            val = float(m[1])
            params[f"PV2_{idx}"] = val
    return params

# proposed
ZPX_PATTERN = re.compile(r"projp(\d+)\s*=\s*(\S+)", re.IGNORECASE)

def extract_zpx_params_new(wat_data):
    params = {}
    for axis, wstr in wat_data.items():
        matches = ZPX_PATTERN.findall(wstr)
        for m in matches:
            idx = int(m[0])
            val = float(m[1])
            params[f"PV2_{idx}"] = val
    return params

# Time old
t0 = time.perf_counter()
for _ in range(1000000):
    extract_zpx_params_old(dummy_data)
t1 = time.perf_counter()

# Time new
t2 = time.perf_counter()
for _ in range(1000000):
    extract_zpx_params_new(dummy_data)
t3 = time.perf_counter()

print(f"Elapsed old: {t1 - t0:.5f}s")
print(f"Elapsed new: {t3 - t2:.5f}s")
