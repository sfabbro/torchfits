import time

def slow_version(paths):
    valid_files = 0
    for path in paths:
        try:
            import os
            if os.path.exists(path):
                valid_files += 1
        except Exception:
            continue
    return valid_files

def fast_version(paths):
    valid_files = 0
    try:
        import os
        for path in paths:
            if os.path.exists(path):
                valid_files += 1
    except Exception:
        pass
    return valid_files

def fast_version_2(paths):
    import os
    valid_files = 0
    for path in paths:
        try:
            if os.path.exists(path):
                valid_files += 1
        except Exception:
            continue
    return valid_files

def run_benchmark():
    paths = [f"path_{i}.fits" for i in range(1000000)]

    start = time.perf_counter()
    slow_version(paths)
    end = time.perf_counter()
    print(f"Slow version taken: {end - start:.6f} seconds")

    start = time.perf_counter()
    fast_version_2(paths)
    end = time.perf_counter()
    print(f"Fast version taken: {end - start:.6f} seconds")

if __name__ == "__main__":
    run_benchmark()
