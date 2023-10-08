import sys
import os
import wrap_integers


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


def mutate_file(input_filename, output_filename):
    with open(input_filename, 'r') as file:
        new_src_code = ""

        for line in file:
            # Process each line here (e.g., wrap integers)
            wrapped = wrap_integers(line)
            
            # Append the processed line (wrapped) to new_src_code
            new_src_code += wrapped

    # After processing all lines, write new_src_code to the output file
    with open(output_filename, 'w') as output_file:
        output_file.write(new_src_code)

    
def run(filename, mutation_num):
    valid_file(filename)
    valid_num(mutation_num)

    mutations = int(mutation_num)
    # print(filename,mutations)

    # write each file out 
    for i in range(mutations):
        # create new src code
        
        new_file_path = "output" + str(i) + ".txt"
        mutate_file(filename,new_file_path)

        print(f"new src code has been written to '{new_file_path}'")

    

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