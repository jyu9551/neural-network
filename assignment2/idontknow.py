import sys
import os

package_dirs = []
for e in sys.path:
  package_dirs.extend([os.path.join(dp, d) for dp, dn, fn in os.walk(e) for d in dn if d])

for d in package_dirs:
  if 'numpy' in d:
    print(d)

#import numpy

def lambda_handler(event, context):
  pass