
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

log_list = """
logs/nodp/20210416/2354/bs16.log
logs/nodp/20210416/2354/bs16_see0.log
logs/nodp/20210416/2354/bs16_seed123.log
logs/nodp/20210416/2354/bs16_seed22.log
logs/nodp/20210416/2354/bs16_seed300.log
logs/partial_dp/20210416/2351/nohidden_lr0.1_norm0.01_sigma0.45
logs/partial_dp/20210416/2351/nohidden_lr0.1_norm0.005_sigma0.45
logs/partial_dp/20210416/2351/nohidden_lr0.1_norm0.01_sigma0.5
logs/partial_dp/20210416/2351/nohidden_lr0.1_norm0.005_sigma0.5
logs/partial_dp/20210416/2351/nohidden_lr0.1_norm0.01_sigma0.55 
logs/partial_dp/20210416/2351/nohidden_lr0.1_norm0.005_sigma0.55 
logs/partial_dp/20210418/1912/nohidden_lr0.1_norm1e-3_sigma0.5
logs/partial_dp/20210418/1912/nohidden_lr0.1_norm5e-4_sigma0.5
logs/partial_dp/20210418/1912/nohidden_lr0.1_norm1e-4_sigma0.5
logs/partial_dp/20210418/1912/nohidden_lr0.1_norm5e-5_sigma0.5
logs/partial_dp/20210418/1912/nohidden_lr0.1_norm1e-5_sigma0.5
logs/partial_dp/20210418/1912/nohidden_lr0.1_norm5e-6_sigma0.5
"""                    
model_paths = extract_model_path(log_list.split('\n'))


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