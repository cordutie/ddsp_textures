import os
import sys

# Add the parent directory to the Python path
parent_dir = os.path.abspath('..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import ddsp_textures.training.wrapper

if __name__ == "__main__":
    # Check if there is only one argument and that is a string
    if len(sys.argv) != 3 or not isinstance(sys.argv[1], str):
        raise NameError("Invalid arguments")
    else:
        method = sys.argv[1]
        path = sys.argv[2] #json path for training and checkpoint parent folder path for retraining
    print("Let's go!")
    if method == "train":
        ddsp_textures.training.wrapper.trainer_SubEnv(path)
    elif method == "retrain":
        ddsp_textures.training.wrapper.trainer_from_checkpoint_SubEnv(path)