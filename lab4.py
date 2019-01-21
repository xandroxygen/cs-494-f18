import sys

from operator import add
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Netflix Ratings").getOrCreate()
user_id = "1488844"


def main():
    print("\n------------------------------------------\n")

    if len(sys.argv) < 2:
        print("Usage <input_file.txt")
        spark.stop()
        exit()

    input_file_name = sys.argv[1]

    lines = spark.read.text(input_file_name).rdd.map(lambda r: r[0])
    same_ranked_users = (
        lines.map(lambda l: l.split("\t"))
        .map(lambda r: (r[1] + ':' + r[2], [r[0]]))
        .reduceByKey(add)
        .filter(lambda r: user_id in r[1])
        .flatMap(lambda tup: [(uid, 1) for uid in tup[1]])
        .reduceByKey(add)
        .filter(lambda u: u[0] != user_id)
        .sortBy(lambda w: w[1], ascending=False)
    )

    print(str(same_ranked_users.take(10)))
    spark.stop()


main()
