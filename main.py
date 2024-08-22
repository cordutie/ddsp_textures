import os
import sys

# Add the parent directory to the Python path
parent_dir = os.path.abspath('..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import ddsp_textures.training.wrapper

if __name__ == "__main__":
    # Check if there is only one argument and that is a string
    if len(sys.argv) != 2 or not isinstance(sys.argv[1], str):
        raise NameError("Invalid arguments")
    else:
        parameters_json_path = sys.argv[1]
    print("Let's go!")
    ddsp_textures.training.wrapper(parameters_json_path)