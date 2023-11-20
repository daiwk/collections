import sys
for line in sys.stdin:
    line = line.strip("\n")
    print(line.replace("XXX", "$$"))
