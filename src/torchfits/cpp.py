import torchfits._C as _C

# Forward all attributes from _C to this module
globals().update({k: v for k, v in _C.__dict__.items() if not k.startswith("__")})
