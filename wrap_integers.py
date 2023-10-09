# This "Starter Code" for EECS 481 HW3 shows how to use a visitor
# pattern to replace nodes in an abstract syntax tree. 
# 
# Note well:
# (1) It does not show you how to read input from a file. 
# (2) It does not show you how to write your resulting source
#       code to a file.
# (3) It does not show you how to "count up" how many of 
#       instances of various node types exist.
# (4) It does not show you how to use random numbers. 
# (5) It does not show you how to only apply a transformation
#       "once" or "some of the time" based on a condition.
# (6) It does not show you how to copy the AST so that each
#       mutant starts from scratch. 
# (7) It does show you how to execute modified code, which is
#       not relevant for this assignment.
#
# ... and so on. It's starter code, not finished code. :-) 
# 
# But it does highlight how to "check" if a node has a particular type, 
# and how to "change" a node to be different. 

import ast
import astor
import random

class MyVisitor(ast.NodeTransformer):
    """Notes all Numbers and all Strings. Replaces all numbers with 481 and
    strings with 'SE'."""
    def __init__(self, seed=None):
        self.count_int = 0
        self.count_str = 0
        self.random_seed = seed  # Store the seed

    # Note how we never say "if node.type == Number" or anything like that.
    # The Visitor Pattern hides that information from us. Instead, we use
    # these visit_Num() functions and the like, which are called
    # automatically for us by the library. 
    def visit_Num(self, node):
        print("Visitor sees a number: ", ast.dump(node), " aka ", astor.to_source(node))
        # Note how we never say "node.contents = 481" or anything like
        # that. We do not directly assign to nodes. Intead, the Visitor
        # Pattern hides that information from us. We use the return value
        # of this function and the new node we return is put in place by
        # the library. 
        # Note: some students may want: return ast.Num(n=481) a
        self.count_int += 1

        # Set the random seed before generating a random number
        if self.random_seed is not None:
            random.seed(self.random_seed)

        if node.n > 50:
            # Replace Num nodes greater than 50 with a constant value
            return ast.Num(value=100, kind=None)
        return ast.Num(value=random.randint(1, 100), kind=None)
        # return ast.Num(value=90, kind=None)

    def visit_Str(self, node):
        print("Visitor sees a string: ", ast.dump(node), " aka ", astor.to_source(node))
        # Note: some students may want: return ast.Str(s=481)
        self.count_str += 1
        # Replace Str nodes with random strings

        # Set the random seed before generating random strings
        if self.random_seed is not None:
            random.seed(self.random_seed)

        result = ""
        stringSize = random.randint(1, 10)
        for i in range(stringSize):
            result += chr(random.randint(97, 122))
            # result += chr(97)
        

        return ast.Str(value=result, kind=None)
    
    def visit_Compare(self, node):
        # Handle comparison operators (e.g., >, ==, <)
        left = self.visit(node.left)
        ops = [self.visit(op) for op in node.ops]
        comparators = [self.visit(comp) for comp in node.comparators]

        # Example: Replace 'x > 50' with 'x < 100'
        if isinstance(node.ops[0], ast.Gt) and isinstance(node.comparators[0], ast.Num):
            # Set the random seed before generating a random number
            if self.random_seed is not None:
                random.seed(self.random_seed)

            # Replace 'x > 50' with 'x < 100'
            new_op = ast.Lt()
            new_comp = ast.Num(value=random.randint(1, 100), kind=None)
            return ast.Compare(left=left, ops=[new_op], comparators=[new_comp])

        return ast.Compare(left=left, ops=ops, comparators=comparators)


def main(line, seed):
    code = line
    tree = ast.parse(code)
    tree = MyVisitor(seed).visit(tree)
    # Add lineno & col_offset to the nodes we created
    ast.fix_missing_locations(tree)
    updated_code = astor.to_source(tree)
    return updated_code


# Instead of reading from a file, the starter code always processes in 
# a small Python expression literally written in this string below: 
# code = """print(111 + len("hello") + 222 + len("goodbye"))"""

# As a sanity check, we'll make sure we're reading the code
# correctly before we do any processing. 


# print("Before any AST transformation")
# print("Code is: ", code)
# print("Code's output is:") 
# exec(code)      # not needed for HW3
# print()

# Now we will apply our transformation. 

# print("Applying AST transformation")
# tree = ast.parse(code)
# tree = MyVisitor().visit(tree)
# # Add lineno & col_offset to the nodes we created
# ast.fix_missing_locations(tree)
# print("Transformed code is: ", astor.to_source(tree))
# co = compile(tree, "", "exec")
# print("Transformed code's output is:") 
# exec(co)        # not needed for HW3