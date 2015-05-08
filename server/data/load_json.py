import json
from pprint import pprint
#with open('3news.data') as data_file:
with open('3news.constraint.json') as data_file:
    data = json.load(data_file)

#print(data['mustlink'])    # data is a dictionary with 2 elements
print(data['mustlink'][0])  # {'target': 360, 'similar': 480}
print(data['mustlink'][0]['target'])
print(data['cannotlink'][5]['target'])
#pprint(data)