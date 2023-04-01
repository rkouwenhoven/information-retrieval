import csv
from pathlib import Path

import numpy as np
from fast_forward.ranking import Ranking
from fast_forward.index import Mode, InMemoryIndex
from fast_forward.encoder import TCTColBERTQueryEncoder as TCTColBERTQueryEncoderFF
from ir_measures import read_trec_qrels, calc_aggregate, nDCG, RR


if __name__=="__main__":

    # Sparse ranking
    sparse_ranking_2019 = Ranking.from_file(Path("msmarco-passage-test2019-sparse10000.txt"))

    # Index
    index = InMemoryIndex.from_disk(
        Path("ffindex_passage_2019_2020.pkl"),
        encoder=TCTColBERTQueryEncoderFF("castorini/tct_colbert-msmarco")
    )

    # Queries 2019
    with open(
            "msmarco-test2019-queries.tsv",
            encoding="utf-8",
            newline=""
    ) as fp:
        queries = {q_id: q for q_id, q in csv.reader(fp, delimiter="\t")}
    print(f"loaded {len(queries)} queries")

    # Ranking 2019
    alpha = 0.2
    result = index.get_scores(
        sparse_ranking_2019,
        queries,
        alpha=alpha,
        cutoff=10,
        early_stopping=False
    )

    qrels = list(read_trec_qrels("2019qrels-pass.txt"))
    print(
        "BM25",
        calc_aggregate([nDCG @ 10, RR(rel=2) @ 10], qrels, sparse_ranking_2019.run)
    )
    print(
        f"BM25, TCTColBERT (alpha={alpha})",
        calc_aggregate([nDCG @ 10, RR(rel=2) @ 10], qrels, result[alpha].run)
    )
