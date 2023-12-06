import re
import sys

f = open(sys.argv[1], 'r+')
s = f.read()
f.close()
newS = ""
for d in re.findall(r'.+', s):
    (th, q, p, tileQ, tileM, k, rp, rc, fused, elem, rowModTileIsZero, kEqVar, dist) = d.split(',')
    newS += ",".join((th, q, p, tileQ, k, tileM, fused, dist, rp, rc, elem, rowModTileIsZero, kEqVar)) + "\n"

print(newS)