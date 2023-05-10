# -*- coding: utf-8 -*-
"""
Created on Wed May 10 10:34:27 2023

@author: Roman A.
"""

print('this is a test')

import time

def test(long int Range):

    cdef unsigned long long int total = 1
    cdef int k
    cdef float t1, t2, t
    
    
    t1 = time.time()
    
    for k in range(1,Range+1):
        total = total * k
    print "Total =", total
    
    t2 = time.time()
    t = t2-t1
    print(t)