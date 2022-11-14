from math import log

def discre(X):
	mx = max(X)
	mn = min(X)
	n = len(X)
	base = (mx-mn)/n
	ret = []
	for i in X:
		tmp = int(i/base)
		tmp = tmp*base
		ret.append(tmp)
	return ret

def H1(in_x):
	x = discre(in_x)
	count={}
	for i in x:
		if(not(i in count.keys())):
			count[i]=0
		count[i]=count[i]+1

	ans=0.0
	tot=0
	for key,value in count.items():
		tot+=value
	for key,value in count.items():
		ans=ans-value/tot*log(value/tot)/log(2)

	return ans

def H(in_X,in_Y):
	'''
	H(X|Y)
	'''
	X = discre(in_X)
	Y = discre(in_Y)
	y={}
	for i in Y:
		if(not (i in y.keys())):
			y[i]=0
		y[i]=y[i]+1

	n=len(X)
	ans=0.0
	for i in y.keys():
		x=[X[j] for j in range(n) if Y[j]==y[i]]
		ans+=H1(x)*y[i]/n
	return ans


X=[1,2,3,1]
Y=[1,1,2,2]

print(H1(X))
print(H(X,Y))