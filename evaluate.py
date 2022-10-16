import os
import csv
import json
import logging
import argparse
from dotenv import load_dotenv
from src.preprocessing import read_json_file
from sklearn.metrics import classification_report

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

def parse_arguments():

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--gold_path', type=str, nargs='?', required=True, help='Path to the gold labels file')
    arg_parser.add_argument('--pred_path', type=str, nargs='?', required=True, help='Path to the predicted labels file')
    arg_parser.add_argument('--out_path', type=str, nargs='?', required=True, help='Path where to save scores')

    arg_parser.add_argument('--summary_exps', type=str, nargs='?', required=True, help='Path to the summary of the overall experiments')

    return arg_parser.parse_args()

def get_metrics(gold_path, predicted_path):

    # setup label types
    label_types = {label: idx for idx, label in enumerate(os.getenv(f"RELATION_LABELS").split())}

    _, _, _, gold_relations = read_json_file(gold_path, label_types, multi_label=True)

    # get the predicted labels
    predicted = []
    with open(predicted_path) as predicted_file:
        predicted_reader = csv.reader(predicted_file, delimiter=',')
        next(predicted_reader)
        for line in predicted_reader:
            pred_instance = [0] * len(label_types.keys())
            for rel in line[0].split(' '):
                pred_instance[label_types[rel]] = 1
            predicted.append(pred_instance)
    
    assert len(gold_relations) == len(predicted), "Length of gold and predicted labels should be equal."

    labels = os.getenv(f"RELATION_LABELS").split()
    report = classification_report(gold_relations, predicted, target_names=labels, output_dict=True, zero_division=0)

    # do not consider the labels with 0 instances in the test set in the macro-f1 computation
    macro = sum([elem[1]['f1-score'] if elem[0] in label_types.keys() and elem[1]['support'] > 0 else 0 for elem in report.items()]) / sum([1 if elem[0] in label_types.keys() and elem[1]['support'] > 0 else 0 for elem in report.items()])

    return report, macro


if __name__ == '__main__':

    args = parse_arguments()

    logging.info(f"Evaluating {args.gold_path} and {args.pred_path}.")

    metrics, macro = get_metrics(args.gold_path, args.pred_path)

    logging.info(f"Saving scores to {args.out_path} -> Macro F1: {macro * 100}")
    exp = os.path.splitext(os.path.basename(args.pred_path))[0]
    json.dump(metrics, open(f"{os.path.join(args.out_path, exp)}-results.json", "w"))

    with open(args.summary_exps, 'a') as file:
        file.write(f"Micro F1: {metrics['micro avg']['f1-score'] * 100}\n")
        file.write(f"Macro F1: {macro * 100}\n")
        file.write(f"Weighted F1: {metrics['weighted avg']['f1-score'] * 100}\n")