import csv
from pathlib import Path

from fast_forward.encoder import TCTColBERTQueryEncoder
from fast_forward.index import InMemoryIndex, Mode
from fast_forward.ranking import Ranking
from ir_measures import read_trec_qrels, calc_aggregate, nDCG, RR

sparse_ranking_2019 = Ranking.from_file(Path("../ff_index/dev/testing/msmarco-passage-test2019-sparse10000.txt"))
sparse_ranking_2019.cut(5000)

with open(
    "../ff_index/dev/testing/msmarco-test2019-queries.tsv",
    encoding="utf-8",
    newline=""
) as fp:
    queries = {q_id: q for q_id, q in csv.reader(fp, delimiter="\t")}
print(f"loaded {len(queries)} queries")

encoder = TCTColBERTQueryEncoder("castorini/tct_colbert-msmarco")
# load an index from disk into memory
index = InMemoryIndex.from_disk(Path("../ff_index/dev/testing/ffindex_passage_2019_2020.pkl"), encoder, Mode.MAXP)

alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
result = index.get_scores(
    sparse_ranking_2019,
    queries,
    alpha=alpha,
    early_stopping=False,
    useCc=True
)

qrels = list(read_trec_qrels("../ff_index/dev/testing/2019qrels-pass.txt"))
print(
    "BM25",
    calc_aggregate([nDCG@10, RR(rel=2)@10], qrels, sparse_ranking_2019.run)
)
for a in alpha:
    print(
        f"BM25, TCTColBERT (alpha={alpha})",
        calc_aggregate([nDCG@10, RR(rel=2)@10], qrels, result[a].run)
    )
