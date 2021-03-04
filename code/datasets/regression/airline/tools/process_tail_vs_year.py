
look_up = {}
for idx,line in enumerate(open('/tmp/plane-data-processed','r')):
	if idx == 0:
		continue

	line_split = line.split("\n")[0].split()

	if len(line_split) == 1 or line_split[1]=="None":
		continue
	look_up[line_split[0]] = int(line_split[1])

key_error = 0
for line in open('/tmp/tail-id','r'):
	idx = line.split("\n")[0]
	try:
		age = 2008.-float(look_up[idx])
		print(age)
	except:
		key_error += 1
		print("NA")
