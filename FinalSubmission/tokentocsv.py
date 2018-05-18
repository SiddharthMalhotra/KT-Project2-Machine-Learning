import sys, re

# Name of the file that contains tokens
tokens_file = 'tokens'

# Token length, default 3
token_length = 3

# Input file
input_file = 'maininput_2.txt'


# Output file
output_file = 'output7.csv'

with open(tokens_file) as f:
    content = f.readlines()
# remove whitespace characters like `\n` at the end of each line
contents = [x.strip() for x in content]

processed_tokens = []
for content in contents:
    words = content.split()
    if len(words) == token_length:
        processed_tokens.append(content)

output_header = 'id'
for processed_token in processed_tokens:
    output_header += ',' + processed_token
output_header += '\n'

## magick starts, calculate frequency of the matched words and write to CSV
with open(input_file) as f:
    content = f.readlines()
input_file_contents = [x.strip() for x in content]

output_content = ''

for input_file_content in input_file_contents:
    current_content = input_file_content.split('\t')
    current_line = str(current_content[0]) + ','
    for processed_token in processed_tokens:
        current_line += str( len(re.findall(processed_token, current_content[2], flags=re.IGNORECASE)) ) + ','
    output_content += current_line.rstrip(",") + "\n"

output_content = output_content.rstrip('\n')

## Writing data to ouput file
op_file = open(output_file, "a+")
op_file.write(output_header)
op_file.write(output_content)
op_file.close()
