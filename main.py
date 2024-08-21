import training.wrapper
import sys

if __name__ == "__main__":
    # Check if there is only one argument and that is a string
    if len(sys.argv) != 2 or not isinstance(sys.argv[1], str):
        raise NameError("Invalid arguments")
    else:
        parameters_json_path = sys.argv[1]
    print("Let's go!")
    training.wrapper.train(parameters_json_path)