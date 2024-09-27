import json
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import os
import numpy as np
import scipy.stats as ss

def rank_correlation(model_results, gt_key, pred_key):
    """
    Calculate the rank correlation between human and predicted errors.
    """
    model_list = model_results.keys()

    models = []
    humans_errors = []
    pred_errors = []
    for model_name in model_list:
        models.append(model_name)
        humans_error, pred_error = np.mean(model_results[model_name][gt_key]), np.mean(model_results[model_name][pred_key])

        humans_errors.append(humans_error)
        pred_errors.append(pred_error)

    estimated_rank = ss.rankdata(np.array(pred_errors))
    human_rank = ss.rankdata(np.array(humans_errors))

    return spearmanr(estimated_rank, human_rank)

def model_corr_result(data, 
                      score_dict,
                      source_name=None,
                      gt_score='faithfulness_score',
                      machine_score='pred_faithfulness_score'):
    """
    Calculate correlation results for the given data.

    data : List of input data.
    score_dict : The dimension of the metric to evaluate.
    source_name : The name of the source.
    unique_id_name : The unique identifier key. Defaults to 'doc_id'.
    machine_score : The name of machine score
    """
    model_results = {}

    # Filter the data for the given source name
    if source_name is not None:
        data = [item for item in data if item['source'] == source_name]

    for item in data:
        model_name = item['model']

        # Initialize model result dictionary if not already present
        if model_name not in model_results:
            model_results[model_name] = {
                'gt_scores': [],
                'pred_machine_scores': [],
            }
    
        model_results[model_name]['gt_scores'].append(item[gt_score])
        model_results[model_name]['pred_machine_scores'].append(item[score_dict][machine_score])

    # Pearson correlation (summary level)
    pearson_corr, peason_corr_pvalue = pearsonr([item[gt_score] for item in data], 
                                                [item[score_dict][machine_score] for item in data])

    print("pearson correlation : ", pearson_corr, peason_corr_pvalue)
    
    # Spearman correlation (summary level)
    spearman_corr, spearman_corr_pvalue = spearmanr([item[gt_score] for item in data],
                                                    [item[score_dict][machine_score] for item in data])

    print("spearman correlation : ", spearman_corr, spearman_corr_pvalue)

    # System level rank correlation
    rank_corr, rank_corr_pvalue = rank_correlation(model_results, 'gt_scores', 'pred_machine_scores')
    print("rank Correlation:", rank_corr, rank_corr_pvalue)

    results = {'source': source_name,
               'pearson_corr': pearson_corr,
               'pearson_corr_pvalue': peason_corr_pvalue,
               'spearman_corr': spearman_corr,
               'spearmancorr_pvalue': spearman_corr_pvalue,
               'rank_corr': rank_corr,
               'rank_corr_pvalue': rank_corr_pvalue,
               'metric': machine_score}

    return results

def export_benchmark_result(annotation_data, 
                             output_path, 
                             score_dict,
                             gt_score):
    """
    Export annotation results to a CSV file.

    annotation_data : The list of input data.
    output_path : The output file path.
    score_dict : The list of the metric to evaluate.

    """
    data_lst = []

    metric_lst = annotation_data[0][score_dict].keys()

    for item in annotation_data:
        if item['source'] not in data_lst:
            data_lst.append(item['source'])
        
    result_list = []

    for dataset_name in data_lst:
        for metric in metric_lst:
            print(dataset_name, ' : ', metric)
            print('-'*100)
            result_list.append(model_corr_result(data=annotation_data,
                                                 source_name=dataset_name,
                                                 score_dict=score_dict,
                                                 gt_score=gt_score,
                                                 machine_score=metric))
            print('-'*100)

    pd.DataFrame(result_list).to_csv(output_path)

    return pd.DataFrame(result_list)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process annotation data.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input JSON file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output CSV file')
    parser.add_argument('--score_dict', type=str, required=True, help='Metric dimension to evaluate')
    parser.add_argument('--gt_score', type=str, required=True, help='Ground truth score')

    args = parser.parse_args()

    with open(args.input_path, 'r') as f:
        _data = [json.loads(line) for line in f]

    _data = [item for item in _data if item['summary_success_state'] == 'success']

    result = export_benchmark_result(annotation_data=_data,
                                      output_path=args.output_path,
                                      score_dict=args.score_dict,
                                      gt_score=args.gt_score,
                                      )