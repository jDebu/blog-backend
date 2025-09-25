from pyspark.sql import SparkSession, functions as F, Window


def main() -> None:
    spark = SparkSession.builder.appName("predecesor_sucesor_demo").getOrCreate()

    # Datos de ejemplo
    predecesores = [
        (1, "P1"),
        (2, "P2"),
        (3, "P3"),
        (4, "P4"),
        (5, "P5"),
    ]

    sucesores = [
        (101, "S1"),
        (102, "S2"),
        (103, "S3"),
    ]

    dfp = spark.createDataFrame(predecesores, ["id", "nombre"])  # predecesores
    dfs = spark.createDataFrame(sucesores, ["id", "nombre"])     # sucesores

    # Índices deterministas por id
    w = Window.orderBy("id")
    dfp = dfp.withColumn("p_idx", F.row_number().over(w))  # 1..N
    dfs = dfs.withColumn("s_idx", F.row_number().over(w))  # 1..M

    # Orden de ejecución
    # - Predecesores en posiciones 1,2,4,5,7,8,... => i + floor((i-1)/2)
    # - Sucesores en posiciones 3,6,9,... => 3*j
    dfp_ord = (
        dfp
        .withColumn("orden", F.col("p_idx") + F.floor((F.col("p_idx") - 1) / 2))
        .withColumn("tipo", F.lit("predecesor"))
        .select("orden", "tipo", "id", "nombre", "p_idx")
    )

    dfs_ord = (
        dfs
        .withColumn("orden", 3 * F.col("s_idx"))
        .withColumn("tipo", F.lit("sucesor"))
        .select("orden", "tipo", "id", "nombre", "s_idx")
    )

    schedule = (
        dfp_ord.select("orden", "tipo", "id", "nombre")
        .unionByName(dfs_ord.select("orden", "tipo", "id", "nombre"))
        .orderBy("orden")
    )

    print("=== Orden de ejecución (simulación) ===")
    schedule.show(truncate=False)

    # Aristas del árbol: cada S_j depende de P_(2j-1) y P_(2j)
    parents_index_a = dfs.select(((2 * F.col("s_idx") - 1)).alias("p_idx"), F.col("id").alias("sucesor_id"))
    parents_index_b = dfs.select((2 * F.col("s_idx")).alias("p_idx"), F.col("id").alias("sucesor_id"))
    parents_index = parents_index_a.unionByName(parents_index_b)

    edges = (
        parents_index
        .join(dfp.select("p_idx", F.col("id").alias("predecesor_id")), on="p_idx", how="left")
        .select("predecesor_id", "sucesor_id")
        .orderBy("sucesor_id", "predecesor_id")
    )

    print("=== Aristas del árbol (predecesor -> sucesor) ===")
    edges.show(truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()

