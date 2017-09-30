import csv

def exportResults(results, csvFile):
    """
    Export dog probability
    :param results:
    :param csvFile:
    :return:
    """
    with open(csvFile, 'w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['id', 'label'])
        for result in results:
            writer.writerow([result[0], result[1]])
