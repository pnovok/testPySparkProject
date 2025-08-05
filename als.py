# # Alternating Least Squares
# 
# This is an example implementation of ALS for learning how to use Spark.
# Please refer to
# [this example](https://github.com/apache/spark/blob/master/examples/src/main/python/ml/als_example.py) # for more conventional use.
# 
# This example requires [NumPy](http://www.numpy.org/)

from __future__ import print_function
import sys
import numpy as np
from numpy.random import rand
from numpy import matrix
from pyspark.sql import SparkSession

LAMBDA = 0.01   # regularization
np.random.seed(42)

def rmse(R, ms, us):
    diff = R - ms * us.T
    return np.sqrt(np.sum(np.power(diff, 2)) / (M * U))

def update(i, vec, mat, ratings):
    uu = mat.shape[0]
    ff = mat.shape[1]

    XtX = mat.T * mat
    Xty = mat.T * ratings[i, :].T

    for j in range(ff):
        XtX[j, j] += LAMBDA * uu

    return np.linalg.solve(XtX, Xty)

spark = SparkSession\
    .builder\
    .appName("PythonALS")\
    .getOrCreate()

sc = spark.sparkContext

M = 100
U = 500
F = 10
ITERATIONS = 5
partitions = 2

print("Running ALS with M=%d, U=%d, F=%d, iters=%d, partitions=%d\n" %
      (M, U, F, ITERATIONS, partitions))

R = matrix(rand(M, F)) * matrix(rand(U, F).T)
ms = matrix(rand(M, F))
us = matrix(rand(U, F))

Rb = sc.broadcast(R)
msb = sc.broadcast(ms)
usb = sc.broadcast(us)

for i in range(ITERATIONS):
    ms = sc.parallelize(range(M), partitions) \
           .map(lambda x: update(x, msb.value[x, :], usb.value, Rb.value)) \
           .collect()
    # collect() returns a list, so array ends up being
    # a 3-d array, we take the first 2 dims for the matrix
    ms = matrix(np.array(ms)[:, :, 0])
    msb = sc.broadcast(ms)

    us = sc.parallelize(range(U), partitions) \
           .map(lambda x: update(x, usb.value[x, :], msb.value, Rb.value.T)) \
           .collect()
    us = matrix(np.array(us)[:, :, 0])
    usb = sc.broadcast(us)

    error = rmse(R, ms, us)
    print("Iteration %d:" % i)
    print("\nRMSE: %5.4f\n" % error)

spark.stop()
