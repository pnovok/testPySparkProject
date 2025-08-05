# # PageRank
# 
# This is an example implementation of PageRank. For more conventional use,
# Please refer to [graphx](http://spark.apache.org/docs/latest/graphx-programming-guide.html#pagerank).

from __future__ import print_function
import re
import sys
from operator import add
from pyspark.sql import SparkSession

def computeContribs(urls, rank):
    """Calculates URL contributions to the rank of other URLs."""
    num_urls = len(urls)
    for url in urls:
        yield (url, rank / num_urls)

def parseNeighbors(urls):
    """Parses a urls pair string into urls pair."""
    parts = re.split(r'\s+', urls)
    return parts[0], parts[1]

print("""WARN: This is a naive implementation of PageRank and is
      given as an example! Please refer to PageRank implementation provided by graphx""",
      file=sys.stderr)

spark = SparkSession\
    .builder\
    .appName("PythonPageRank")\
    .getOrCreate()

# Loads in input file. It should be in format of:
# ```
#     URL         neighbor URL
#     URL         neighbor URL
#     URL         neighbor URL
#     ...
# ```
!hdfs dfs -put resources/data/mllib/pagerank_data.txt /tmp
lines = spark.read.text("/tmp/pagerank_data.txt").rdd.map(lambda r: r[0])

# Loads all URLs from input file and initialize their neighbors.
links = lines.map(lambda urls: parseNeighbors(urls)).distinct().groupByKey().cache()

# Loads all URLs with other URL(s) link to from input file and initialize ranks of them to one.
ranks = links.map(lambda url_neighbors: (url_neighbors[0], 1.0))

# Calculates and updates URL ranks continuously using PageRank algorithm.
for iteration in range(5):
    # Calculates URL contributions to the rank of other URLs.
    contribs = links.join(ranks).flatMap(
        lambda url_urls_rank: computeContribs(url_urls_rank[1][0], url_urls_rank[1][1]))

    # Re-calculates URL ranks based on neighbor contributions.
    ranks = contribs.reduceByKey(add).mapValues(lambda rank: rank * 0.85 + 0.15)

# Collects all URL ranks and dump them to console.
for (link, rank) in ranks.collect():
    print("%s has rank: %s." % (link, rank))

spark.stop()

