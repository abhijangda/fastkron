from .fastkronbase import fastkronX86, fastkronCUDA

if fastkronX86 is not None and fastkronCUDA is not None:
    __version__ = fastkronX86.version() + "+" + fastkronCUDA.version()
elif fastkronX86 is not None:
    __version__ = fastkronX86.version()
elif fastkronCUDA is not None:
    __version__ = fastkronCUDA.version()
else:
    __version__ = "1.0+shuffle"
