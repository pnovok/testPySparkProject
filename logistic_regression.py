# # Logistic regression
#
# A logistic regression implementation that uses [NumPy](http://www.numpy.org)
# to act on batches of input data using efficient matrix operations.
# 
# In practice, one may prefer to use the LogisticRegression algorithm in
# ML, as shown in 
# [this example](https://github.com/apache/spark/blob/master/examples/src/main/python/ml/linear_regression_with_elastic_net.py).

from __future__ import print_function
import sys
import numpy as np
from pyspark.sql import SparkSession

D = 10  # Number of dimensions

# Read a batch of points from the input file into a NumPy matrix object. We operate on batches to
# make further computations faster.
# The data file contains lines of the form <label> <x1> <x2> ... <xD>. We load each block of these
# into a NumPy array of size numLines * (D + 1) and pull out column 0 vs the others in gradient().
def readPointBatch(iterator):
    strs = list(iterator)
    matrix = np.zeros((len(strs), D + 1))
    for i, s in enumerate(strs):
        matrix[i] = np.fromstring(s.replace(',', ' '), dtype=np.float32, sep=' ')
    return [matrix]

spark = SparkSession\
    .builder\
    .appName("PythonLR")\
    .getOrCreate()

# Add the data file to hdfs.
!hdfs dfs -put resources/data/mllib/lr_data.txt /tmp

points = spark.read.text("/tmp/lr_data.txt").rdd.map(lambda r: r[0])\
    .mapPartitions(readPointBatch).cache()
iterations = 10

# Initialize w to a random value
w = 2 * np.random.ranf(size=D) - 1
print("Initial w: " + str(w))

# Compute logistic regression gradient for a matrix of data points
def gradient(matrix, w):
    Y = matrix[:, 0]    # point labels (first column of input file)
    X = matrix[:, 1:]   # point coordinates
    # For each point (x, y), compute gradient function, then sum these up
    return ((1.0 / (1.0 + np.exp(-Y * X.dot(w))) - 1.0) * Y * X.T).sum(1)

def add(x, y):
    x += y
    return x

for i in range(iterations):
    print("On iteration %i" % (i + 1))
    w -= points.map(lambda m: gradient(m, w)).reduce(add)

print("Final w: " + str(w))

spark.stop()
