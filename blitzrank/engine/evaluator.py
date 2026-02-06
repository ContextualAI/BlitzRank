from typing import Dict, Any, Tuple
from pathlib import Path
from .config import EvaluationConfig

import tempfile
import os
import pytrec_eval


def trec_eval(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: Tuple[int] = (10, 50, 100, 200, 1000),
) -> Dict[str, float]:
    ndcg, _map, recall = {}, {}, {}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {map_string, ndcg_string, recall_string}
    )
    scores = evaluator.evaluate(results)

    for query_id in scores:
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]

    def _normalize(m: dict) -> dict:
        return {k: round(v / len(scores), 5) for k, v in m.items()}

    ndcg = _normalize(ndcg)
    _map = _normalize(_map)
    recall = _normalize(recall)

    all_metrics = {}
    for mt in [ndcg, _map, recall]:
        all_metrics.update(mt)

    return all_metrics


def remove_duplicate(response):
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
        else:
            print("duplicate")
    return new_response


def clean_response(response: str):
    new_response = ""
    for c in response:
        if not c.isdigit():
            new_response += " "
        else:
            new_response += c
    new_response = new_response.strip()
    return new_response


class EvalFunction:
    @staticmethod
    def write_file(rank_results, file):
        print("write_file")
        with open(file, "w") as f:
            for i in range(len(rank_results)):
                rank = 1
                hits = rank_results[i]["hits"]
                for hit in hits:
                    # Sanitize docid to replace spaces with underscores for TREC format compatibility
                    sanitized_docid = str(hit["docid"]).replace(" ", "_")
                    f.write(
                        f"{hit['qid']} Q0 {sanitized_docid} {rank} {hit['score']} rank\n"
                    )
                    rank += 1
        return True

    @staticmethod
    def write_qrels_file(qrels: Dict[str, Dict[str, int]]) -> str:
        temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".qrels")
        for qid, docs in qrels.items():
            for docid, rel in docs.items():
                # Sanitize docid to replace spaces with underscores for TREC format compatibility
                sanitized_docid = str(docid).replace(" ", "_")
                temp_file.write(f"{qid}\t0\t{sanitized_docid}\t{rel}\n")
        temp_file.close()
        return temp_file.name

    @staticmethod
    def main(args_qrel, args_run):
        assert os.path.exists(args_qrel)
        assert os.path.exists(args_run)

        with open(args_qrel, "r") as f_qrel:
            qrel = pytrec_eval.parse_qrel(f_qrel)

        with open(args_run, "r") as f_run:
            run = pytrec_eval.parse_run(f_run)

        all_metrics = trec_eval(qrel, run, k_values=(1, 5, 10))
        print(all_metrics)
        return all_metrics


def evaluate_results(
    dataset: Dict[str, Any], config: EvaluationConfig, output_dir: Path, logger
) -> Dict[str, float]:
    if config.type == "none":
        logger.info("Evaluation skipped")
        return {}

    elif config.type == "trec_eval":
        qrels = dataset["qrels"]

        trec_file = output_dir / "results.trec"
        if trec_file.exists():
            logger.info(f"Using existing TREC file: {trec_file}")
        else:
            results = dataset["results"]
            logger.info(f"Writing TREC file to: {trec_file}")
            EvalFunction.write_file(results, str(trec_file))

        logger.info("Running trec_eval")
        qrels_file = EvalFunction.write_qrels_file(qrels)
        metrics = EvalFunction.main(qrels_file, str(trec_file))

        return metrics

    else:
        raise ValueError(f"Unknown evaluation type: {config.type}")
