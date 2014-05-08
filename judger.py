"""Judges inputs.
"""
import sys

def main():
    instance, solution = sys.argv[1:]

    #Get input
    with open(instance) as f:
        #Get number of vertices, equations, and prime
        V, E, P = [int(x) for x in f.readline().split()]
        equations = []
        #Process each equation
        for line in f:
            equations.append([int(x) for x in line.split()])

    with open(solution) as f:
        assignment = [int(x) for x in f.readline().split()]
        assert len(assignment) == V
        satisfied = 0

        for equation in equations:
            a, b, c, d, e = equation
            assert b != d
            assert 1 <= a and 1 <= c
            assert a <= P - 1 and c <= P - 1
            assert 0 <= e and e <= P - 1

            if (a * assignment[b] + c * assignment[d] + e) % P == 0:
                satisfied += 1

        print "%s equations satisfied." % satisfied

if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv = ['', 'test.in', 'solution.out']
    main()