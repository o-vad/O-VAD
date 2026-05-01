import subprocess
import sys
import argparse
import os
import os.path as osp

sys.path.insert(0, osp.dirname(osp.dirname(__file__)))  # add proj dir to path
from utils import load_yaml_file

def run_evaluation(config, pred, anno):
    """Run both evaluation scripts and tally results"""
    
    # Run temporal localization evaluation
    print(f"Running temporal localization evaluation for {pred}...")
    temploc_cmd = ["python3", "eval/compute_temploc_pr.py", "-c", config, "-p", pred, "-a", anno]
    temploc_result = subprocess.run(temploc_cmd)
    
    if temploc_result.returncode != 0:
        print(f"Error running temploc evaluation: {temploc_result.stderr}")
        return None
    
    # Run semantic accuracy evaluation
    print(f"Running semantic accuracy evaluation for {pred}...")
    sem_cmd = ["python3", "eval/compute_sem_acc.py", "-c", config, "-p", pred, "-a", anno]
    sem_result = subprocess.run(sem_cmd)
    
    if sem_result.returncode != 0:
        print(f"Error running semantic evaluation: {sem_result.stderr}")
        return None
    
    cfg = load_yaml_file(config)

    with open(osp.join(cfg.paths.evaldir, f'temporal_loc_{pred}.txt'), 'r') as fh:
        temploc_results = [x.strip() for x in fh.readlines()]

    with open(osp.join(cfg.paths.evaldir, f'sem_acc_{pred}.txt'), 'r') as fh:
        sem_results = [x.strip() for x in fh.readlines()]

    result = {
        'temploc_pre': float(temploc_results[1].split(',')[1]) * 100,
        'temploc_rec': float(temploc_results[1].split(',')[2]) * 100,
        'semacc_verb': float(sem_results[0].split(':')[-1]) * 100,
        'semacc_obj': float(sem_results[2].split(':')[-1]) * 100,
        'tf_recall_sem_agnostic': float(sem_results[4].split(':')[-1]) * 100,
        'tf_recall': float(sem_results[6].split(':')[-1]) * 100,
    }

    return result

def main():
    parser = argparse.ArgumentParser(description='Evaluate state-graph performances')
    parser.add_argument('-c', '--config', required=True, help='Configuration file path')
    parser.add_argument('-p', '--pred', required=True, help='Prediction identifier')
    parser.add_argument('-a', '--anno', type=str, default='VOST-TAS/vost_tas.json', help='annotation path')
    
    args = parser.parse_args()

    print(f"Evaluating state-graph performances for config: {args.config}, pred: {args.pred}, anno: {args.anno}")
    print("=" * 60)

    results = run_evaluation(args.config, args.pred, args.anno)
    if results:
        headers = ['Sem-Acc Verb', 'Sem-Acc Obj', 'Temp-Loc Pre', 'Temp-Loc Rec', 'TF Recall Sem Agnostic', 'TF Recall']
        header_str = ' &'.join(['      '] + headers)
        perf_str = '       &' + ' &'.join(['{:.1f}'.format(results[h.lower().replace("-", "").replace(" ", "_")]) for h in headers])
        print(header_str)
        print(perf_str)
        with open(osp.join(load_yaml_file(args.config).paths.evaldir, f'_FINAL_state_graph_{args.pred}.txt'), 'w') as f:
            f.write(header_str + '\n')
            f.write(perf_str + '\n')

    else:
        print("Failed to get evaluation results")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())