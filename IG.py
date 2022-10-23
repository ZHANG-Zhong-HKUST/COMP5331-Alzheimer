from math import log
def H1(x):
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

def H(X,Y):
	'''
	H(X|Y)
	'''
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