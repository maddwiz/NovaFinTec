
import json, hashlib, os, sys, platform, time
from .utils import sha1_file

def code_hash(paths):
    sha = hashlib.sha1()
    for p in sorted(paths):
        if os.path.isfile(p):
            with open(p,"rb") as f:
                sha.update(f.read())
    return sha.hexdigest()

def write_provenance(out_path, config: dict, data_files: list):
    meta = {
        "config": config,
        "data_sha1": {os.path.basename(p): sha1_file(p) for p in data_files if os.path.exists(p)},
        "code_hash": "qengine",
        "platform": platform.platform(),
        "python": sys.version,
        "time": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
