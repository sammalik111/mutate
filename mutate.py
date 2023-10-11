import sys
import os
import hashlib
import ast
import astor
import random

class MyVisitor(ast.NodeTransformer):
    """Notes all Numbers and all Strings. Replaces all numbers with 481 and
    strings with 'SE'."""
    def __init__(self, seed=None):
        self.count_int = 0.0
        self.count_str = 0.0
        self.count_binop = 0.0
        self.count_comp = 0.0
        self.random_seed = seed  # Store the seed
        self.probAdder = 0.5


    
    def visit_Num(self, node):
        print("Visitor sees a number: {}, aka {}".format(ast.dump(node), astor.to_source(node)))
        self.count_int += self.probAdder
        probability = self.count_int / 100
        actual = random.random()
        if actual < probability:
            self.count_int = 0
            if self.random_seed is not None:
                random.seed(self.random_seed)

            # Introduce mutations that are more likely to affect the code behavior
            if node.n > 0:
                # Reduce the number by 1
                return ast.Num(n=node.n - 1)
            else:
                # Increase the number by 1
                return ast.Num(n=node.n + 1)
        else:
            return ast.Num(node.n)


    def visit_Str(self, node):
        print("Visitor sees a string: {}, aka {}".format(ast.dump(node), astor.to_source(node)))
        self.count_str += self.probAdder
        probability = self.count_str / 100
        actual = random.random()
        if actual < probability:
            self.count_str = 0
            if self.random_seed is not None:
                random.seed(self.random_seed)

            # Introduce mutations that are more likely to affect the code behavior
            if node.s:
                # Change the first character of the string
                new_string = chr(random.randint(97, 122)) + node.s[1:]
                return ast.Str(s=new_string)
            else:
                # Generate a completely new string
                result = ""
                string_size = random.randint(0, 10)
                for i in range(string_size):
                    result += chr(random.randint(97, 122))
                return ast.Str(s=result)
        else:
            return ast.Str(node.s)


    def visit_Compare(self, node):
        self.count_comp += self.probAdder
        proability = self.count_comp / 100
        actual = random.random()
        if actual < proability:
            for i in range(len(node.ops)):
                if isinstance(node.ops[i], ast.GtE):
                    self.count_comp = 0
                    node.ops[i] = ast.Lt()
                elif isinstance(node.ops[i], ast.LtE):
                    self.count_comp = 0
                    node.ops[i] = ast.Gt()
                elif isinstance(node.ops[i], ast.Lt):
                    self.count_comp = 0
                    node.ops[i] = ast.GtE()
                elif isinstance(node.ops[i], ast.Gt):
                    self.count_comp = 0
                    node.ops[i] = ast.LtE()
                # elif isinstance(node.ops[i], ast.Eq):
                #     self.count_comp = 0
                #     node.ops[i] = ast.NotEq()
            return self.generic_visit(node)
        else:
            return self.generic_visit(node)

    def visit_BinOp(self, node):
        self.count_binop += self.probAdder
        probability = self.count_binop / 100
        actual = random.random()

        if actual < probability:
            self.count_binop = 0
            new_op = None

            if isinstance(node.op, ast.Add):
                new_op = ast.Sub()
            elif isinstance(node.op, ast.Sub):
                new_op = ast.Add()
            elif isinstance(node.op, ast.Mult):
                new_op = ast.FloorDiv()
                # Change the right operand to 1/right for multiplication
                node.right = ast.BinOp(left=ast.Num(n=1), op=ast.Div(), right=node.right)
            elif isinstance(node.op, ast.FloorDiv):
                new_op = ast.Mult()
                # Change the right operand to 1/right for division
                node.right = ast.BinOp(left=ast.Num(n=1), op=ast.Div(), right=node.right)

            if new_op:
                # Create a new BinOp node with the modified operator
                new_node = ast.BinOp(left=node.left, op=new_op, right=node.right)

                # Preserve the logical value of the comparison
                if isinstance(node.op, ast.Add) or isinstance(node.op, ast.Sub):
                    # If the original operator was Add or Sub, negate the right operand
                    new_node.right = ast.UnaryOp(op=ast.USub(), operand=node.right)

                return new_node

        return self.generic_visit(node)



def valid_file(filename):
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

def mutate_file(input_filename, output_filename, seed_offset):
    with open(input_filename, 'r') as file:
        input_code = file.read()
        code = input_code
        tree = ast.parse(code)
        tree = MyVisitor(seed_offset).visit(tree)
        ast.fix_missing_locations(tree)
        updated_code = astor.to_source(tree)

    with open(output_filename, 'w') as output_file:
        if updated_code == "NONE":
            output_file.write(input_code)
        output_file.write(updated_code)

def run(filename, mutation_num):
    valid_file(filename)
    valid_num(mutation_num)
    mutations = int(mutation_num)

    arguements = filename + mutation_num
    hashed_seed = int(hashlib.sha256(arguements.encode()).hexdigest(), 16)

    for i in range(mutations):
        new_file_path = str(i) + ".py"
        mutate_file(filename, new_file_path, hashed_seed + i)
        print("new src code has been written to '{}'".format(new_file_path))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Invalid number of arguments!\n"
              "Expected usage: python LocalTest.py TESTFILE NUM_MUTATIONS\n"
              "For example: python mutate.py testfile.py 5\n")
    else:
        filename = sys.argv[1]
        mutation_num = sys.argv[2]
        run(filename, mutation_num)