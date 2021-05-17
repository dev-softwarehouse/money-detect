import csv
import glob

with open('dataset_file_names.csv', 'w') as f:
    writer = csv.writer(f)
    a = glob.glob('dataset/*.jpg')
    writer.writerows(zip(a))
