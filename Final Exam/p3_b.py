#' ## Part B - Infeasible Netwon Method
def r_d(x,nu):
	s1 = np.dot(np.transpose(A),nu)
	s2 = grad_f(x).reshape(n,1)
	return s1+s2

def r_p(x):
	return (np.dot(A,x)-b).reshape(p,1)

def r_objective():
	s1 = np.zeros(p*p).reshape(p,p)
	s2 = np.concatenate((np.transpose(A),s1))
	s3 = np.concatenate((hess_f(x),A))
	return np.concatenate((s3,s2),axis=1)

def r(x,nu):
	s1 = r_d(x,nu)
	s2 = r_p(x)
	s3 = -np.concatenate((s1,s2))
	s4 = r_objective()
	return solve(s4,s3)

def get_r(x,nu):
	R = r(x,nu)
	xnt = R[0:n]
	nnt = R[n:n+p]
	return xnt, nnt


def infeasible_newton_method(x,iterations,alpha,beta,eps):
	# Repeat
	y = np.array([])
	s = np.array([])

	nu = np.zeros(p)
	nu = nu.reshape(p,1)

	hx = inv(hess_f(x))
	gx = grad_f(x)

	xnt = -np.dot(hx,gx)
	dec = -np.dot(np.transpose(gx),xnt)

	p = f(x)
	y = np.append(y,p)


	return y,s,p


nu = np.zeros((p,1))
y = np.array([])
y = np.append(y,f(x))

while np.not_equal(np.dot(A,x),b).all() or norm(r(x,nu))>eps:

	xnt,nnt =  get_r(x,nu)

	t=1

	while norm(r(x+t*xnt,nu+t*nnt))>(1-alpha*t)*norm(r(x,nu)):
		t=beta*t

	x += t*xnt
	x = x + abs(np.ones((len(x),1))*min(x))+0.00001
	nu += t*nnt
	y = np.append(y,f(x))

print(y)