from pyspark import SparkContext
from operator import add
import re

sc = SparkContext("local", "Lab3")

top_words = (
    sc.textFile("bible.txt")
    .flatMap(lambda l: l.split(" "))
    .map(lambda w: re.sub(r'[^a-zA-Z-]+', '', w).lower())
    .filter(lambda w: w != '' and not w.isspace())
    .map(lambda w: (w, 1))
    .reduceByKey(add)
    .sortBy(lambda w: w[1], ascending=False)
    .take(20)
)

file = open("lab3.txt", "w")
file.write(str(top_words))
file.close()
