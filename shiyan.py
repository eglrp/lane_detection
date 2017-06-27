l = [1,2,3,5,6,11,23,24,25,26]
result = []
for i in range(len(l)):
	l[i] = l[i]*2
i = 0
while i < len(l):
	ele = []
	step = 0
	while (l[i]+step*2) in l:
		ele.append(l[i+step])
		step = step + 1
	# 这里把ele插入返回的列表中
	result.append(ele)
	i = i + step
print(result)
