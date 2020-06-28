#!/usr/bin/env python
# -*- coding: gbk -*-
########################################################################
# 
########################################################################
 
"""
File: trans_format.py
Date: 2019/02/10 17:36:03
"""
import sys

for line in sys.stdin:
    line = line.strip("\n")
    if line.lstrip(" ").startswith("- ["):
        continue
    print(line.\
            replace("align*", "aligned").\
            replace("../assets/", "./assets/").\
            replace("<html>", "<div>").\
            replace("<br/>", "").\
            replace("</html>", "</div>").\
            replace("## ", "# ").\
            replace("<img src='", "![](").\
            replace("`\(", "$").\
            replace("\)`", "$").\
            replace("`\[", "$$").\
            replace("\]`", "$$").\
            replace("' style='max-height: ", "){ max-height=").\
            replace("px'/>", "px }")
            )
#            replace("`\(", "$").\
#            replace("\)`", "$").\
#            replace("`\[", "$$").\
#            replace("\]`", "$$").\
#
