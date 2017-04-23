import re

files = open('mon.csv','r')
lines = files.readlines()
file2 = open('mon_diff.txt','w')
for line in lines:
	line = line.split(',')
	l = len(line)
	#line = [x for x in line if x !='']
	#line = ','.join(line)
	#line.replace(' ',',').replace('  ',',').replace('   ',',')
	#re.sub('[ ]+',',', line)
	#file2.write(line)
	#file2.write('\n')
files.close()
file2.close()
