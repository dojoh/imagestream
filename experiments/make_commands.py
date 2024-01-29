from pathlib import Path

parts = [
    'bsub  -R "tensorcore" -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=2G -L /bin/bash -q gpu "source ./start_plankton.sh && python main.py --config=./configs/temperature_experiment/alloutliers/aug/',
    '";',
]

config_folder = "/home/o340n/projects/fileserver/cluster-home/projects/2023-konstanz/configs/temperature_experiment/alloutliers/aug"

configs = Path(config_folder).rglob("*.json")

for config in configs:
    print(parts[0] + config.name + parts[1])
