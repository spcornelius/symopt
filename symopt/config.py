__all__ = []
__all__.extend([
    'HAS_IPOPT'
])

try:
    import ipopt
    HAS_IPOPT = True
except ImportError:
    HAS_IPOPT = False
