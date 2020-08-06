
import zlib

_compression_scheme_code_map = {0: 'zlib', 1: 'zstandard', 2: 'lz4', 3: 'snappy', 4: 'bzip', 5: 'lzma', 6: 'blosc',
                                7: 'blosc', 8: 'blosc', 9: 'blosc', 10: 'blosc', 11: 'blosc'}

_import_states = {'zlib': True, 'blosc': True, 'zstd': True, 'snappy': True, 'lz4': True, 'lzma': True, 'bz2': True}

try:
    import blosc
except ImportError:
    _import_states['blosc'] = False

try:
    import zstandard as zstd
except ImportError:
    _import_states['zstandard'] = False

try:
    import snappy
except ImportError:
    _import_states['snappy'] = False

try:
    import lz4.frame
except ImportError:
    _import_states['lz4'] = False

try:
    import lzma
except ImportError:
    _import_states['lzma'] = False

try:
    import bz2
except ImportError:
    _import_states['bz2'] = False


def de_compress(compression_scheme, compressed_data, decompressor_context):

    if compression_scheme == 0:     # zlib
        return zlib.decompress(compressed_data)

    elif compression_scheme == 1:   # zstd
        return decompressor_context.decompress(compressed_data)

    elif compression_scheme == 2:   # lz4
        return lz4.frame.decompress(compressed_data)

    elif compression_scheme == 3:   # snappy
        return snappy.decompress(compressed_data)

    elif compression_scheme == 4:   # bzip
        return bz2.decompress(compressed_data)

    elif compression_scheme == 5:   # lzma
        return lzma.decompress(compressed_data)

    elif compression_scheme == 6:   # blosc_zlib
        return blosc.decompress(compressed_data, as_bytearray=True)

    elif compression_scheme == 7:   # blosc_zstd
        return blosc.decompress(compressed_data, as_bytearray=True)

    elif compression_scheme == 8:   # blosc_lz4
        return blosc.decompress(compressed_data, as_bytearray=True)

    elif compression_scheme == 9:   # blosc_snappy
        return blosc.decompress(compressed_data, as_bytearray=True)

    elif compression_scheme == 10:  # blosclz
        return blosc.decompress(compressed_data, as_bytearray=True)

    elif compression_scheme == 11:  # blosc_lz4hc
        return blosc.decompress(compressed_data, as_bytearray=True)

    else:
        raise NotImplementedError('compression scheme not implemented')


def compress(compression_scheme, compression_level, data, compressor_context):

    if compression_scheme == 0:  # zlib
        return zlib.compress(data, compression_level)

    elif compression_scheme == 1:  # zstd
        return compressor_context.compress(data)

    elif compression_scheme == 2:  # lz4
        return lz4.frame.compress(data, compression_level=compression_level, store_size=False)

    elif compression_scheme == 3:  # snappy
        return snappy.compress(data)

    elif compression_scheme == 4:  # bzip
        return bz2.compress(data, compresslevel=compression_level)

    elif compression_scheme == 5:  # lzma
        return lzma.compress(data, preset=compression_level)

    elif compression_scheme == 6:  # blosc_zlib
        return blosc.compress(data, clevel=compression_level, cname='zlib', shuffle=blosc.BITSHUFFLE)

    elif compression_scheme == 7:  # blosc_zstd
        return blosc.compress(data, clevel=compression_level, cname='zstd', shuffle=blosc.BITSHUFFLE)

    elif compression_scheme == 8:  # blosc_lz4
        return blosc.compress(data, clevel=compression_level, cname='lz4', shuffle=blosc.BITSHUFFLE)

    elif compression_scheme == 9:  # blosc_snappy
        return blosc.compress(data, clevel=compression_level, cname='snappy', shuffle=blosc.BITSHUFFLE)

    elif compression_scheme == 10:  # blosclz
        return blosc.compress(data, clevel=compression_level, cname='blosclz', shuffle=blosc.BITSHUFFLE)

    elif compression_scheme == 11:  # blosc_lz4hc
        return blosc.compress(data, clevel=compression_level, cname='lz4hc', shuffle=blosc.BITSHUFFLE)
    else:
        raise NotImplementedError('compression scheme not implemented')


def import_checks(header):
    if _import_states[_compression_scheme_code_map[header['compression_scheme']]]:
        return True
    else:
        print("For compression code " + str(header['compression_scheme']) +
              " package " + _compression_scheme_code_map[header['compression_scheme']] + " is required.")
        raise ImportError()
