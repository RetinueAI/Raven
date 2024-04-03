import os
from random import sample

import pandas as pd

dfs = []

for i in range(len(os.listdir('./data'))):
    with open(f'data/Raven Dataset Label {i}.csv', 'r') as f:
        lines = f.readlines()

    data = []

    for line in lines:

        if line[-1] == '\n':
            line = line[:-1]

        if line[-1] not in ['.', '?']:
            data.append(line + '.')
        else:
            data.append(line)

    df = pd.DataFrame(data, columns=['inputs'])

    label = 0
    match i:
        case 5|6|7:
            label = 5
        case 8|9|10|11|12|13|14|15:
            label = i - 2
        case _:
            label = i

    df['labels'] = [label for _ in range(len(df))]
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

while True:
    split_char = ""
    end_char = ""
    new_inputs = []

    for i in range(len(df)):
        input = df.iloc[i]['inputs']

        for char in input:
            if char == ".":
                if split_char == "":
                    split_char = "."
                else: 
                    end_char = "."
                    break
            elif char == "?":
                if split_char == "":
                    split_char = "?"
                else:
                    end_char = "?"
                    break


        if len(end_char) > 0:
            uno, dos = input.split(split_char, 1)
            new_inputs.append((uno + dos))
        else:
            new_inputs.append(input)

        split_char = ""
        end_char = ""

    for i in range(len(new_inputs)):
        if '"' in new_inputs[i]:
            sentence = ""
            for part in new_inputs[i].split('"'):
                sentence += part
            new_inputs[i] = sentence
                
        
    df['inputs'] = new_inputs

    split_char = ""
    end_char = ""
    n_shitty = 0

    for i in range(len(df)):
        input = df.iloc[i]['inputs']

        for char in input:
            if char == ".":
                if split_char == "":
                    split_char = "."
                else: 
                    end_char = "."
                    break
            elif char == "?":
                if split_char == "":
                    split_char = "?"
                else:
                    end_char = "?"
                    break


        if len(end_char) > 0:
            n_shitty += 1

        split_char = ""
        end_char = ""

    if n_shitty == 0:
        break

n_labels = len(df.labels.unique())

with open('config.txt', 'w') as f:
    f.write(f"Number of labels:\n")
    f.write(f'{n_labels}')

inputs = {}

for i in range(n_labels):
    inputs[i] = []

for index, row in df.iterrows():
    inputs[row['labels']].append(row['inputs'])

n = min([len(inputs[i]) for i in range(len(inputs))])
print(f'Number of entries from each label is: {n}')
indexes = {}

for i in range(n_labels):
    indexes[i] = {}

new_data = {
    'inputs': [],
    'labels': [],
}

for i in range(len(inputs)):
    for j in range(len(inputs)):
        indexes[i][j] = sample(range(len(inputs[j])), n)   

n_labels = len(inputs)
for label in range(n_labels):
    for i in range(len(indexes[label][0])):
        start = inputs[label][indexes[label][label][i]]
        for j in range(n_labels):
            if label != j:
                new_data['inputs'].append(f'{start} {inputs[j][indexes[label][j][i]]}')
                new_data['labels'].append([label, j])

df = pd.DataFrame(new_data)

input_column = []
output_column = []
_output = []
split_char = ""
end_char = ""

for i in range(len(df)):

    for char in df.iloc[i]['inputs']:
        if char == ".":
            if split_char == "":
                split_char = "."
            else: 
                end_char = "."
                break
        elif char == "?":
            if split_char == "":
                split_char = "?"
            else:
                end_char = "?"
                break


    first, second = df.iloc[i]['inputs'].split(split_char, 1)
    if end_char != "":
        second = second.lower().split(end_char)[0]
    else:
        second = second.lower()

    if second[0:1] != ' ':
        second = ' ' + second

    sentence = (first.lower() + second).strip()
    input_column.append(sentence)
            
    labels = df.iloc[i]['labels']

    for j in range(len(first.split(' '))):
        _output.append(labels[0])
    for j in range(len(second.strip().split(' '))):
        _output.append(labels[1])
    output_column.append(_output)
    _output = []

    split_char = ""
    end_char = ""

df = pd.DataFrame({'inputs': input_column, 'labels': output_column})

incorrect = 0

for i in range(len(df)):
    if len(df.iloc[i]['inputs'].split(' ')) != len(df.iloc[i]['labels']):
        print(df.iloc[i]['inputs'])
        incorrect += 1

if incorrect == 0:
    print("The inputs and outputs matches!")
    df.to_csv('__raven.csv', index=False)
else:
    print("The inputs and outputs don't match up!")