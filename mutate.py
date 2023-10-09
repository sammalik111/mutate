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
        self.count_int = 0
        self.count_str = 0
        self.count_binop = 0
        self.count_comp = 0
        self.random_seed = seed  # Store the seed



    def visit_Num(self, node):
        print("Visitor sees a number: {}, aka {}".format(ast.dump(node), astor.to_source(node)))
        self.count_int += 10
        proability = self.count_int / 100
        actual = random.random()
        if actual < proability:
            self.count_int = 0
            if self.random_seed is not None:
                random.seed(self.random_seed)

            if node.n > 10:
                return ast.Num(n=3)
            return ast.Num(n=random.randint(0, 10))
        else:
            return ast.Num(node.n)

    def visit_Str(self, node):
        print("Visitor sees a string: {}, aka {}".format(ast.dump(node), astor.to_source(node)))
        self.count_str += 10
        proability = self.count_str / 100
        actual = random.random()
        if actual < proability:
            self.count_str = 0
            if self.random_seed is not None:
                random.seed(self.random_seed)

            result = ""
            string_size = random.randint(0, 10)
            for i in range(string_size):
                result += chr(random.randint(97, 122))
            return ast.Str(s=result)
        else:
            return ast.Str(node.s)

    def visit_Compare(self, node):
        self.count_comp += 10
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
                elif isinstance(node.ops[i], ast.Eq):
                    self.count_comp = 0
                    node.ops[i] = ast.NotEq()
            return self.generic_visit(node)
        else:
            return self.generic_visit(node)

    def visit_BinOp(self, node):
        self.count_binop += 10
        proability = self.count_binop / 100
        actual = random.random()
        if actual < proability:
            if isinstance(node.op, ast.Add):
                self.count_binop = 0
                node.op = ast.Sub()
            elif isinstance(node.op, ast.Sub):
                self.count_binop = 0
                node.op = ast.Add()
            elif isinstance(node.op, ast.Mult):
                self.count_binop = 0
                node.op = ast.FloorDiv()
            elif isinstance(node.op, ast.FloorDiv):
                self.count_binop = 0
                node.op = ast.Mult()
            return self.generic_visit(node)
        else:
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
