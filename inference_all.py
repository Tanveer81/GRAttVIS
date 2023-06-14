import os
import sys
import glob
from pathlib import Path
import zipfile
#path_model = '/home/wiss/koner/GenVIS/output/past_query_valid_inst_wdoutMem'
# absolute path to search all text files inside a specific folder
path = f'{sys.argv[1]}/*.pth'
model = sys.argv[1].split('/')[-1]
files = glob.glob(path)
files = [f for f in files if 'model_final' not in f]
files.sort(reverse=True)
#print(files)
for file in files:
    print(file)
    zip_path = f'{sys.argv[1]}/result_zip/{model+"_"+file[-11:-4]}.zip'
    if os.path.exists(zip_path):
        print("Results are already processed for this checkpoint")
        continue
    output_dir = f'{sys.argv[1]}/{file[-11:-4]}'
    os.system(f'srun -u --gres=gpu:{sys.argv[2]} -c {sys.argv[3]} python3 train_net_genvis.py --num-gpus {sys.argv[2]} '
              f' --config-file {sys.argv[1]}/config.yaml --eval-only OUTPUT_DIR {output_dir} '
              f'MODEL.WEIGHTS {file} DATALOADER.NUM_WORKERS 0')
    json_path = f'{output_dir}/inference/results.json'
    Path(f'{sys.argv[1]}/result_zip').mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'w',
                         compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as zf:
        zf.write(json_path, arcname='results.json')
    os.remove(json_path)
