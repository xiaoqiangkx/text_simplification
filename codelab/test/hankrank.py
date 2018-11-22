# Q1
# filename  = ''
#
# from collections import defaultdict
#
# # Count the frequence
# counter = defaultdict(int)
# for line in open(filename):
#     items = line.split(' ')
#     hostname = items[0]
#     counter[hostname] += 1
#
# # Output
# outputs = []
# for hostname in counter:
#     output_line = '%s %s' % (hostname, str(counter[hostname]))
#     outputs.append(output_line)
# output_filename = 'records_%s' % filename
# f = open(output_filename, 'w')
# f.write('\n'.join(outputs))
# f.close()

# Q2
#!/bin/python3

import math
import os
import random
import re
import sys



#!/bin/python3

import math
import os
import random
import re
import sys



#
# Complete the 'missingWords' function below.
#
# The function is expected to return a STRING_ARRAY.
# The function accepts following parameters:
#  1. STRING s
#  2. STRING t
#
from collections import defaultdict

def missingWords_1(s, t):
    t_counter = defaultdict(int)
    for wd in t.split():
        t_counter[wd] += 1

    output = []
    for wd in s.split():
        if wd in t_counter:
            t_counter[wd] -= 1
            if t_counter[wd] < 0:
                output.append(wd)
        else:
            output.append(wd)
    return output

def missingWords_2(s, t):
    s = s.split()
    t = t.split()

    marker = [False] * len(s)
    i, j = 0, 0
    while j < len(t):
        while i < len(s) and j < len(t) and s[i] == t[j]:
            marker[i] = True
            i += 1
            j += 1
        i += 1

    res = []
    for i in range(len(s)):
        if not marker[i]:
            res.append(s[i])
    return res


if __name__ == '__main__':
    s = 'a c a b a a'
    t = 'b a a'
    res1 = missingWords_1(s, t)
    print(res1)
    res2 = missingWords_2(s, t)
    print(res2)