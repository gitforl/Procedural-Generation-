
import matplotlib.pyplot as plt
import numpy as np

import json



folderPath = '../Procedural-Generation-/images/json/rici/'

fileSpecifier = 'combined_pairwise'

fileName = ''#'histogram_'
fileName += fileSpecifier
fileName += '.json'

f = open(folderPath + fileName)
  
data = json.load(f)
  
data = data["binValues"]
  
f.close()

# data = data[:12]

# bins = list(range(0,1024,64))

width = 131072

count = 16

span = width * count
# width = 64

bins = list(range(0,span,width))

# plt.subplots(figsize=(10,5))

plt.bar(bins, data, width = width, align="edge")
plt.xticks(bins, rotation = 45)

# plt.ylim(top = 75000)


plt.show()