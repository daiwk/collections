import sys
for line in sys.stdin:
    x = line.strip("\n").split("S")[1]
    print(x)
    print(len(x))
