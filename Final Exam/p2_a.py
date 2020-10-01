#' ##Problem 2 Take Home Final Exam

# Importing the necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

np.random.seed(999) # Seed random number generator for consistency

#Initialize the matrix dimensions
n = 100
m = 200

# Create a random problem
A = np.random.randn(m,n)

# Pick the stopping condition
eta = 1e-4
i_max = 1000

#' Here I am defining the functions

# Function goes here
def f(x):
	return -sum(np.log(1-np.dot(A,x)))-sum(np.log(1-x))-sum(np.log(1+x))

# Have to differentiate and hard code derivitive
def grad_f(x):
	s1 = 1/(1-np.dot(A,x))
	s2 = np.dot(np.transpose(A),s1)
	return s2+1/(1-x)-1/(1+x)

#' ### Algorithm for Backtracking Line Search
#' while $f(x + t\delta x) > f(x) + \alpha t\gradient f(x) \delta x$: $t = \beta t$

# Backtracking method for updating t
def backtrack(x,alpha,beta):
	dx = -grad_f(x)
	t=1.0
	# Make sure updating to feasible x, use approximation to ensure in feasible domain
	s = x+np.dot(t,dx)
	while (np.amax(np.dot(A,s))>1) or (np.amax(abs(s))>1):
		t=beta*t
		s = x+t*dx
	# Backtrack in feasible domain
	s2 = np.dot(np.transpose(grad_f(x)),dx)
	while f(s) > f(x)+alpha*t*s2:
		t=beta*t
		s = x+t*dx
		s2 = np.dot(np.transpose(grad_f(x)),dx)
	return t

#' ### Algorithm for gradient descent
#' 1. Dx = -1 * gradient f(x)
#' 2. Line search - choose step size t via backtracking method
#' 3. Update x = x + t Dx


def gradient_descent(x,iterations,alpha,beta):
	y = np.array([])
	s = np.array([])
	x = x.reshape(len(x),1) # Make sure x is a vector
	# Repeat
	for i in range(iterations):
		# Step 1
		y = np.append(y,f(x))
		dx = -1*grad_f(x)

		# Step 2
		t = backtrack(x,alpha,beta)
		s = np.append(s,t)

		# Step 3
		x = x + t*dx

		p = f(x)

		if norm(grad_f(x)) < eta:
			break

	return y,s,p

def gradient_descent_plot(alpha,beta):
	# Initialize x(0)
	x = np.zeros(n)

	# Run gradient descent algorithm
	y,s,p = gradient_descent(x,i_max,alpha,beta)

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

# Pick alpha and beta here for backtracking params
alpha = 0.25
beta = 0.5

fig,ax = gradient_descent_plot(alpha,beta)
plt.savefig('Gradient_Descent-fig1.png')

plt.show()





