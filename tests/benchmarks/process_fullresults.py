import sys
import re

def slurp(filepath):
  with open(filepath, 'r') as f:
    return f.read()

def process(filepath):
  for line in re.findall('.+', slurp(filepath)):
    split = line.split('&')
    if len(split) == 5 and float(split[4].strip()) <= 1 and float(split[4].strip()) >= 0 and float(split[1].strip()) != 1.00:
      print(split)



if __name__ == "__main__":
  process(sys.argv[1])