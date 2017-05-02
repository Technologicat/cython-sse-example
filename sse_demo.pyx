# -*- coding: utf-8 -*-
"""Simple example for embedding SSE2 assembly in Cython projects.

The purpose of this project is to provide documentation, since this information was a bit hard to find.

Library module.
"""

from __future__ import division, print_function, absolute_import


# http://stackoverflow.com/questions/11228855/header-files-for-x86-simd-intrinsics
#
# MMX:     mmintrin.h
# SSE:    xmmintrin.h
# SSE2:   emmintrin.h
# SSE3:   pmmintrin.h
# SSSE3:  tmmintrin.h
# SSE4.1: smmintrin.h
# SSE4.2: nmmintrin.h
# SSE4A:  ammintrin.h
# AES:    wmmintrin.h
# AVX:    immintrin.h
# AVX512: zmmintrin.h
#
cdef extern from "emmintrin.h":  # in this example, we use SSE2
    # Two things happen here:
    # - this definition tells Cython that, at the abstraction level of the Cython language, __m128d "behaves like a double"
    # - at the C level, the "cdef extern from" (above) makes the generated C code look up the exact definition from the original header
    #
    ctypedef double __m128d

    # Declare any needed extern functions here; consult $(locate emmintrin.h) and SSE assembly documentation.
    #
    # For example, to pack an (unaligned) double pair, to perform addition and multiplication (on packed pairs),
    # and to unpack the result, one would need the following:
    #
    __m128d _mm_loadu_pd (double *__P) nogil  # (__P[0], __P[1]) are the original pair of doubles
    __m128d _mm_add_pd (__m128d __A, __m128d __B) nogil
    __m128d _mm_mul_pd (__m128d __A, __m128d __B) nogil
    void _mm_store_pd (double *__P, __m128d __A) nogil  # result written to (__P[0], __P[1])


# Simple example: addition.
#
cdef void example1():
    cdef double[2] data
    cdef __m128d mdata, mtmp
    cdef double[2] out

    # cythonic idiom
    #    http://stackoverflow.com/questions/25974975/cython-c-array-initialization
    #
    data[:] = [1.0, 3.0]

    with nogil:
        # pack double pairs into __m128d
        mdata = _mm_loadu_pd( &data[0] )

        # add:  tmp = data + data
        mtmp = _mm_add_pd( mdata, mdata )

        # unpack result
        _mm_store_pd( &out[0], mtmp )

    print( " in", data[0], data[1], sep=", " )  # "1.0, 3.0"
    print( "out", out[0],  out[1],  sep=", " )  # "2.0, 6.0"


# Another example, implementing the following loop,
# of a form that is commonly encountered in ODE integrators:
#
# for j in range(n):
#     out[j] = w1[j] + dt/2 * w2[j]
#
# However, note that this may be slower than plain GCC with "-march=native -msse -msse2 -mfpmath=sse -O2".
# (Tested on Intel Core i5 540M - manually coded version seems about 2% slower.)
#
cdef void example2():
    DEF n = 6       # total number of items to process (must be even)
    DEF m = n // 2  # number of __m128d pairs needed for the items

    cdef double dt = 1.0
    cdef double[2] dthalf
    cdef __m128d mdthalf
    dthalf[:] = [dt*0.5, dt*0.5]

    cdef __m128d mw1, mw2, mtmp
    cdef double[n] w1
    cdef double[n] w2
    cdef double[n] out
    cdef int j
    with nogil:
        mdthalf = _mm_loadu_pd( &dthalf[0] )

        # fill in some test data
        for j in range(n):
            w1[j] = 2.0
            w2[j] = 3.0

        for j in range(m):  # note "m"; each pair contains two items
            mw1  = _mm_loadu_pd( &w1[2*j] )
            mw2  = _mm_loadu_pd( &w2[2*j] )
            mtmp = _mm_mul_pd( mw2, mdthalf )
            mtmp = _mm_add_pd( mw1, mtmp )
            _mm_store_pd( &out[2*j], mtmp )

    print( "result:" )
    for j in range(n):
        print( out[j] )  # each component should be 3.5


def run():
    example1()
    example2()

