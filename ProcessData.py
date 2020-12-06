import glob
import os
import unicodedata
import string
import re


def findFiles(path): 
    return glob.glob(path)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().split('\n')
    return lines

labels = ['injured_or_dead_people','missing,_trapped,_or_found_people','displaced_people_and_evacuations','infrastructure_and_utilities_damage','donation_needs_or_offers_or_volunteering_services','caution_and_advice','sympathy_and_emotional_support','other_useful_information']
all_letters = string.ascii_letters + " .,;'"
routes = findFiles('CrisisNLP/*/*.tsv')
print (routes)


newFile= open("proccessedText2.txt",'a+',encoding="utf-8")
onelabels=0
cerolabels=0

for i in routes:
    lines = readLines(i)
    for j in lines[1:]:
        line = j.split('	')
        try:
            line[1] = re.sub(r'http\S+', '', line[1])
            line[1] = re.sub(r'@\S+', '', line[1])
            if(line[2] in labels):
                line[2] = 1
                onelabels=onelabels+1
            else:
                cerolabels=cerolabels+1
                line[2] = 0
            newFile.write(str(unicodeToAscii(line[1]))+'\t'+str(line[2])+'\t'+ ((i.split('\\'))[1].split('_'))[1]+'\t'+ ((i.split('\\'))[1].split('_'))[2] +'\n')
        except IndexError:
            print('Ready: '+i)

print(onelabels)
print(cerolabels)
exit()