import glob
import re
import subprocess
import os


def main():
    files = glob.glob('train_results/*')
    for pathname in files:
        filename = os.path.basename(pathname)
        dirname = os.path.dirname(pathname)
        match = re.match('(\w+)_indoor_(\d+)step_(\d+)in_(\d+)hidden_(\d+)_(\d+)epoch_(.+)', filename)
        if match is None:
            print(filename)
            pass
        else:
            model = match.group(1)
            num_step = match.group(2)
            in_dim = match.group(3)
            num_hidden = match.group(4)
            date_str = match.group(5)
            num_epoch = match.group(6)
            exp_name = match.group(7)
            new_name = "%s_indoor_%s_%02dstep_%02din_%03dhidden_%03depoch_%s" \
                    % (model, date_str, int(num_step), int(in_dim),
                       int(num_hidden), int(num_epoch), exp_name)
            new_path = os.path.join(dirname, new_name)
            result = subprocess.run(['git', 'mv', pathname, new_path], stdout=subprocess.PIPE)
            pass
        pass
    return


if __name__ == '__main__':
    main()
    pass
