bad_words = ['N']

with open('train1.txt') as oldfile, open('checktoken2.txt', 'w') as newfile:
#    for line in oldfile:
#        line[6] = 'W'
#        newfile.write(x)
    for line in oldfile:
        if not any(bad_word in line for bad_word in bad_words):
            #line.replace('Y'," ")
            newfile.write(line)
