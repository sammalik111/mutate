import sys


def test():
    pass

# Use this block of code to run the test via a command line
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Invalid number of arguments!\n Expected usage: python LocalTest.py TESTFILE TESTNUM\n"
              "For example: python mutate.py testfile.py (number_of_mutations)")
    else:
        filename = sys.argv[1]
        mutation_num = sys.argv[2]
        test()