import csv
import re

with open('ECG_Interpretation_New.csv', 'r', encoding="utf-8-sig") as csvinput:
    with open('output.csv', 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        reader = csv.reader(csvinput)

        all = []
        for row in reader: 
            sample_num = row[0]
            
            # Get model results - plurality 2
            found = False
            f1 = open('plurality2ModelResults.txt', 'r')
            for line in f1:
                sample = re.search('Sample: (\d{4}-\d+) ', line, re.IGNORECASE)
                if sample != None: 
                    if sample.group(1) == sample_num:
                        model = re.search('Function: (\d?), (\d?)', line, re.IGNORECASE)
                        if model != None: 
                            row.append(model.group(1))
                            row.append(model.group(2))
                            found = True 
                            break
            
            if not found: 
                # Get model results - plurality 3
                f2 = open('plurality3ModelResults.txt', 'r')
                for line in f2:
                    sample = re.search('Sample: (\d{4}-\d+) ', line, re.IGNORECASE)
                    if sample != None: 
                        if sample.group(1) == sample_num:
                            model = re.search('Function: (\d?), (\d?)', line, re.IGNORECASE)
                            if model != None: 
                                row.append(model.group(1))
                                row.append(model.group(2))
                                break
            
            f1.close()
            f2.close()
            writer.writerow(row)
