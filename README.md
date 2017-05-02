# cython-sse

Simple example for embedding SSE2 assembly in Cython projects

## Introduction

As it says on the tin.

The purpose of this project is to provide documentation, since this information was a bit hard to find.

Be aware that in the case of a simple loop over data, performing the same manipulation on each array element, GCC with the compiler flags `-march=native -mfpmath=sse -msse -msse2 -O2` is pretty good at generating SSE code. Manually tuned SSE for such trivial tasks might actually be slower (and less maintainable, and specific to a CPU architecture) than just writing (sensible) plain C and letting GCC take care of the performance optimization.

Only more advanced algorithms designed specifically for low-level vectorization are likely to see significant (or indeed any) benefit.

If you still wish to access SSE manually, read on.

## The __m128d datatype

Defining the `__m128d` datatype in Cython is the tricky part. The answer can be found [in this thread on cython-users](https://groups.google.com/forum/#!topic/cython-users/zpT61irJcqA) (search terms used: "cython typedef sse").

Cython needs to be told, within the constraints of its syntax, that `__m128d` *behaves like a* `double`.

Then, the C compiler needs the exact definition. To get the Cython-generated C code to `#include` it, the `ctypedef` must be `cdef extern from`'d from the original header.

See [sse_demo.pyx](sse_demo.pyx).

## Further reading

General notes on SSE in Cython can be found in [this cython-users thread](https://groups.google.com/forum/#!msg/cython-users/LfBH6M7gNTc/B19uFB5YbYYJ) (search terms: "cython sse").

If the SSE part can be isolated in its own C file, there is [another approach](https://xor0110.wordpress.com/2010/09/16/using-the-sse-rsqrt-from-python-via-cython/) that can be used.

On SSE in general:

 - [Header files for SIMD intrinsics](http://stackoverflow.com/questions/11228855/header-files-for-simd-intrinsics): `mmintrin.h`, `xmmintrin.h`, `emmintrin.h`, ...
 - [SSE for GCC](http://stackoverflow.com/questions/661338/sse-sse2-and-sse3-for-gnu-c)
 - [C++ - Getting started with SSE](http://felix.abecassis.me/2011/09/cpp-getting-started-with-sse/)

## License

[The Unlicense](LICENSE.md)

