

randn('state',1);
m=200;
n=100;
ALPHA = 0.01;
BETA = 0.5;
MAXITERS = 1000;
NTTOL = 1e-8;
GRADTOL = 1e-3;


% generate random problem
A = randn(m,n);
% gradient method
vals = []; steps = [];
x = zeros(n,1);

for iter = 1:MAXITERS
    val = -sum(log(1-A*x)) - sum(log(1+x)) - sum(log(1-x));
    vals = [vals, val];
    d = 1./(1-A*x);
    grad = A'*d - 1./(1+x) + 1./(1-x);
    v = -grad;
    fprime = grad'*v;
    
    norm(grad);
    if norm(grad) < GRADTOL, break; end;
    
    t = 1;
    while ((max(A*(x+t*v)) >= 1) | (max(abs(x+t*v)) >= 1)),
        t = BETA*t;
    end;
    
    while ( -sum(log(1-A*(x+t*v))) - sum(log(1-(x+t*v).^2)) > ...
        val + ALPHA*t*fprime )
        t = BETA*t;
    end;
    
    x = x+t*v;
    
    steps = [steps,t];
end;
optval = vals(length(vals));
figure(1)
semilogy([0:(length(vals)-2)], vals(1:length(vals)-1)-optval, '-');
xlabel('x'); ylabel('z');
figure(2)
plot([1:length(steps)], steps, ':',[1:length(steps)], steps, 'o');
xlabel('x'); ylabel('z');


% Newton method
vals = []; steps = [];
x = 0.5*ones(n,1);
for iter = 1:MAXITERS
%     val = -sum(log(1-A*x)) - sum(log(1+x)) - sum(log(1-x));
    val = x'*log(x);
    vals = [vals, val];
%     d = 1./(1-A*x);
%     grad = A'*d - 1./(1+x) + 1./(1-x);
    grad = log(x) + ones(length(x));
%     hess = A'*diag(d.^2)*A + diag(1./(1+x).^2 + 1./(1-x).^2);
    hess = diag(1/x);
    v = -inv(hess)*grad;
    fprime = grad'*v;
    if abs(fprime) < NTTOL, break; end;
    t = 1;
    while ((max(A*(x+t*v)) >= 1) | (max(abs(x+t*v)) >= 1)),
        t = BETA*t;
    end;
    while ( -sum(log(1-A*(x+t*v))) - sum(log(1-(x+t*v).^2)) > ...
        val + ALPHA*t*fprime )
        t = BETA*t;
    end;
    x = x+t*v;
    steps = [steps,t];
end;
optval = vals(length(vals));
figure(3)
semilogy([0:(length(vals)-2)], vals(1:length(vals)-1)-optval, '-', ...
    [0:(length(vals)-2)], vals(1:length(vals)-1)-optval, 'o');
xlabel('x'); ylabel('z');
figure(4)
plot([1:length(steps)], steps, '-', [1:length(steps)], steps, 'o');
axis([0, length(steps), 0, 1.1]);
xlabel('x'); ylabel('z');