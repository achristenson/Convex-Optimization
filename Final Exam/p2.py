#' #Problem 2 Take Home Final Exam

# Importing the necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm, inv
from scipy.optimize import line_search

np.random.seed(999) # Seed random number generator for consistency

#Initialize the matrix dimensions
n = 100
m = 200

# Create a random problem
A = np.random.randn(m,n)

# Pick the stopping condition
eta = 1e-4
eps = 1e-8
i_max = 1000

#' ## Part a - Gradient Descent

# Function goes here
def f(x):
	s1 = 1-np.dot(A,x)
	s2 = np.log(s1)
	s3 = np.log(1-x)
	s4 = np.log(1+x)
	s5 = -sum(s2)-sum(s3)-sum(s4)
	return s5

# Have to differentiate and hard code derivitive
def grad_f(x):
	s1 = 1/(1-np.dot(A,x))
	s2 = np.dot(np.transpose(A),s1)
	return s2+1/(1-x)-1/(1+x)

#' ### Algorithm for Backtracking Line Search
#' while f(x + t delta x) > f(x) + alpha t gradient(f(x)) delta x: t = beta t

# Backtracking method for updating t
def backtrack(x,f,grad_f,dx,alpha,beta):
	t=1.0
	y = f(x)
	gx = grad_f(x)
	while np.amax(np.dot(A,(x+t*dx)))>=1 or np.amax(abs(x+t*dx)>=1) :
		t = beta*t
	while f(x+t*dx) > y+alpha*t*np.dot(np.transpose(gx),dx):
		t=beta*t
	return t



#' ### Algorithm for gradient descent
#' 1. Dx = -1 * gradient f(x)
#' 2. Line search - choose step size t via backtracking method
#' 3. Update x = x + t Dx


def gradient_descent(iterations,alpha,beta):
	y = np.array([])
	s = np.array([])
	# Use zero as initial guess
	x = np.zeros(n)
	x = x.reshape(n,1) # Make sure x is a vector
	# Repeat
	for i in range(iterations):
		# Step 1
		y = np.append(y,f(x))
		dx = -1*grad_f(x)

		# Step 2
		t = backtrack(x,f,grad_f,dx,alpha,beta)

		s = np.append(s,t)

		# Step 3
		x = x + t*dx

		p = f(x)

		if norm(grad_f(x)) < eta:
			break
	return y,s,p

#' Making a plot function to play with alpha and beta more easily

def gradient_descent_plot(alpha,beta):
	# Run gradient descent algorithm
	y,s,p = gradient_descent(i_max,alpha,beta)

	fig,ax = plt.subplots(2,1)

	ax[0].semilogy(y-p)
	ax[0].set_title('Convergence to Optimal Solution')
	ax[0].set_xlabel('Iterations')
	ax[0].set_ylabel('f - p_star')

	ax[1].stem(s)
	ax[1].set_title('Step Length')
	ax[1].set_xlabel('Iterations')
	ax[1].set_ylabel('t')

	fig.suptitle('Gradient Descent, alpha = %1.2f, beta = %1.2f'%(alpha,beta),fontsize=16)

	return fig,ax

#' ### Figures and analysis

#' Below is the code to perform gradient descent with alpha=0.25, beta=0.5

# fig,ax = gradient_descent_plot(alpha,beta)
fig0,ax0 = gradient_descent_plot(0.25,0.5)
plt.savefig('fig1.png')

#' Below is the code to perform gradient descent with alpha=0.1, beta=0.5

fig1,ax1 = gradient_descent_plot(0.01,0.5)
plt.savefig('fig2.png')

#' The decrease in alpha eliminates some of the larger step size changes, and yields a smoother but slightly less accurate convergence in the bounds of the iterations
#' Let's try playing with beta a little bit now

fig2,ax2 = gradient_descent_plot(0.01,0.1)
plt.savefig('fig3.png')

#' Great! Our convergence has gotten even more accurate. Notice also the step size increase.

#' ## Part b - repeat using netwon's method

def hess_f(x):
	s1 = 1/(1-np.dot(A,x))
	s2 = np.diag(np.power(s1,2)[:,0])
	s3 = np.dot(s2,A)
	s4 = np.dot(np.transpose(A),s3)
	s5 = np.power(1+x,2)
	s6 = np.diag(1/s5)
	s7 = np.power(1-x,2)
	s8 = np.diag(1/s7)
	return s4+s6+s8

def newton_method(x,iterations,alpha,beta,eps):
	# Repeat
	y = np.array([])
	s = np.array([])

	hx = inv(hess_f(x))
	gx = grad_f(x)

	dnt = -np.dot(hx,gx)
	dec = -np.dot(np.transpose(gx),dnt)

	p = f(x)
	y = np.append(y,p)

	for i in range(iterations):

		if dec/2 <= eps:
			break

		t = backtrack(x,f,grad_f,dnt,alpha,beta)
		s = np.append(s,t)

		x = x + t*dnt

		p = f(x)
		y = np.append(y,p)

		hx = inv(hess_f(x))
		gx = grad_f(x)

		dnt = -np.dot(hx,gx)
		dec = -np.dot(np.transpose(gx),dnt)

	return y,s,p

def newton_method_plot(alpha,beta,eps):
	# Run gradient descent algorithm
	x = np.zeros(n)
	x = x.reshape(n,1)
	y,s,p = newton_method(x,i_max,alpha,beta,eps)

	fig,ax = plt.subplots(2,1)

	ax[0].semilogy(y-p)
	ax[0].set_title('Convergence to Optimal Solution')
	ax[0].set_xlabel('Iterations')
	ax[0].set_ylabel('f - p_star')

	ax[1].stem(s)
	ax[1].set_title('Step Length')
	ax[1].set_xlabel('Iterations')
	ax[1].set_ylabel('t')

	fig.suptitle('Newton Method, alpha = %1.2f, beta = %1.2f'%(alpha,beta),fontsize=16)

	return fig,ax

#' ### Figures and analysis

#' Below is the code to perform newton method with alpha=0.25, beta=0.5

#fig,ax = newton_method_plot(alpha,beta)
fig3,ax3 = newton_method_plot(0.25,0.5,eps)
plt.savefig('fig4.png')

#' Below is the code to perform newton with alpha=0.1, beta=0.5

fig4,ax4 = newton_method_plot(0.01,0.5,eps)
plt.savefig('fig5.png')

#' Let's try playing with beta a little bit now

fig5,ax5 = newton_method_plot(0.01,0.1,eps)
plt.savefig('fig6.png')

#' Notice how in comparison to the gradient descent method, the newton method converges much more quickly than the gradient descent
#' After doing problem three I noticed the newton method for this problem takes way longer to converge. I think this is because the backtracking method that I use here dramatically slows down the convergence rate because it forces the step sizes to be smaller than desireable.

