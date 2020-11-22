srun -p p -t 2999:99 --gres=gpu:1 --pty python3 createModel.py -g 0
srun -p p -t 2700:00 --gres=gpu:1 --pty python3 emoji_output.py -g 0
