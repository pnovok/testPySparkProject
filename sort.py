# # Sorting
#
# Sorting key-value pairs by key.

from __future__ import print_function
import sys
from pyspark.sql import SparkSession

spark = SparkSession\
    .builder\
    .appName("PythonSort")\
    .getOrCreate()

# Add the data file to hdfs.
!hdfs dfs -put resources/kv1.txt /tmp

lines = spark.read.text("/tmp/kv1.txt").rdd.map(lambda r: r[0])
sortedCount = lines.flatMap(lambda x: x.split(' ')[0]) \
    .map(lambda x: (int(x), 1)) \
    .sortByKey()

# This is just a demo on how to bring all the sorted data back to a single node.
# In reality, we wouldn't want to collect all the data to the driver node.
output = sortedCount.collect()
for (num, unitcount) in output:
    print(num)

spark.stop()
