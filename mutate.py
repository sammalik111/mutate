import sys
import os

# ERROR CHECKING #
def valid_file(filename):
    # Check if the file exists in the current directory
    if os.path.exists(filename):
        pass
    else:
        print("Invalid filename inputted!\n"
              "Expected usage: python LocalTest.py TESTFILE NUM_MUTATIONS\n"
              "For example: python mutate.py testfile.py 5\n")
        exit()

def valid_num(mutation_num):
    if mutation_num.isdigit() and int(mutation_num) > 0:
        pass
    else:
        print("Invalid number of mutations specified!\n"
              "Expected usage: python LocalTest.py TESTFILE NUM_MUTATIONS\n"
              "For example: python mutate.py testfile.py 5\n")
        exit()
# ERROR CHECKING #




def run(filename, mutation_num):
    valid_file(filename)
    valid_num(mutation_num)

    mutations = int(mutation_num)
    print(filename,mutations)
    

# Use this block of code to run the test via a command line
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Invalid number of arguments!\n"
              "Expected usage: python LocalTest.py TESTFILE NUM_MUTATIONS\n"
              "For example: python mutate.py testfile.py 5\n")
    else:
        filename = sys.argv[1]
        mutation_num = sys.argv[2]
        run(filename,mutation_num)