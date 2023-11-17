import sys
for line in sys.stdin:
    line = line.strip("\n")
    if "![]" in line:
        line = line.split("[](")[1].split(')')[0]
        #print '<img src="%s" title="A dummy picture" alt="The gif" />' % line
        #print '<img href="%s" />' % line
        print line + " "
        continue
    print line
