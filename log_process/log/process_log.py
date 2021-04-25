
import os
import re

def extract_model_path(log_list):
    model_paths = []
    seeds = []
    for l in log_list:
        if l:
            with open(l.strip(), 'r') as fh:
                lines = fh.readlines()
                for i, line in enumerate(lines):
                    if line.startswith('seed: '):
                        seeds.append(line.split(": ")[1].strip('\n'))
                    if line.startswith('**********'):
                        model_path = '/'.join(lines[i+1].split('/')[:4])
                        model_paths.append(model_path)
                        print(model_path)
                        break
    return list(zip(model_paths, seeds))

def extract_model_test_ppl(log_list):
    test_ppls = []
    test_accs = []
    for l in log_list:
        if l:
            with open(l.strip(), 'r') as fh:
                lines = fh.readlines()
                for i, line in enumerate(lines):
                    if "End of training" in line:
                        # matched = re.search("test ppl   (.+?) | test acc (.+?)", line)
                        # test_ppl, test_acc = matched.group(1), matched.group(2)
                        test_ppl, test_acc = line.split("test ppl")[1].split("|")[0].strip(), line.split("test acc")[1].strip("\n").strip()
                        test_ppls.append(test_ppl)
                        test_accs.append(test_acc)
    print("test ppl")
    for p in test_ppls:
        print(p)
    print("test acc")
    for p in test_accs:
        print(p)
    return list(zip(test_ppls, test_accs))

log_list = """
logs/partial_dp/20210423/nohidden_lr0.1_norm5e-3_sigma0.1_seed1111 
logs/partial_dp/20210423/nohidden_lr0.1_norm0.25_sigma0.1_seed1111  
logs/partial_dp/20210423/nohidden_lr0.1_norm5e-3_sigma0.05_seed1111  
logs/partial_dp/20210423/nohidden_lr0.1_norm0.25_sigma0.05_seed1111  
logs/partial_dp/20210423/nohidden_lr0.1_norm5e-3_sigma0.01_seed1111"""                    

model_paths = extract_model_path(log_list.split('\n'))

test_results = extract_model_test_ppl(log_list.split('\n'))

def print_canary_attack_command(model_paths):
    """
    python attacks/canary_insertion.py -bs 256 --checkpoint model/dp/20210414/002943/ --cuda cuda:0 --outputf attacks/canary_insertion/dp_sigma05_lr01_norm002_seed0.csv
    """
    outputs = []
    for p, s in model_paths:
        sample_path = os.listdir(p)[0]
        output = p.split('/')[1]
        if output != 'nodp':
            matched = re.search('_lr-(.+?)_.+_sigma-(.+?)_norm-(.+?)_', sample_path)
            lr, sigma, norm = matched.group(1), matched.group(2), matched.group(3)
            output = f"attacks/canary_insertion/{output}/lr{lr}_sigma{sigma}_norm{norm}_seed{s}.csv"
        else:
            output = f"attacks/canary_insertion/{output}/nodp_seed{s}.csv"
        outputs.append(output)
        canary_command = f'python attacks/canary_insertion.py -bs 256 --checkpoint {p} --cuda cuda:0 --outputf {output}'
        print(canary_command)
    assert len(outputs) == len(set(outputs)), f"repetitive paths, {outputs}, {set(outputs)}"

    for o in outputs:
        assert not os.path.exists(o), f"{o} exists!"

def print_membership_attack_command(model_paths):
    """
    'python attacks/mem_inference.py -ckpt model/dp/20210409/223157/ --outputf attacks/membership_inference/dp_lr005_sigma05_norm01_1000.csv --cuda cuda:5 --N 1000 -bs 64'
    """
    outputs = []
    for p, s in model_paths:
        sample_path = os.listdir(p)[0]
        output = p.split('/')[1]
        if output != 'nodp':
            matched = re.search('_lr-(.+?)_.+_sigma-(.+?)_norm-(.+?)_', sample_path)
            lr, sigma, norm = matched.group(1), matched.group(2), matched.group(3)
            output = f"attacks/membership_inference/{output}/lr{lr}_sigma{sigma}_norm{norm}_seed{s}.csv"
        else:
            output = f"attacks/membership_inference/{output}/nodp_seed{s}.csv"
        outputs.append(output)
        canary_command = f'python attacks/mem_inference.py -bs 64 --N 1000 --checkpoint {p} --cuda cuda:0 --outputf {output}'
        print(canary_command)
    assert len(outputs) == len(set(outputs)), f"repetitive paths, {outputs}, {set(outputs)}"

    for o in outputs:
        assert not os.path.exists(o), f"{o} exists!"

print_canary_attack_command(model_paths)
print("=============================")
print("membership ")

print_membership_attack_command(model_paths)