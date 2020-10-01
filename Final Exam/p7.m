clear all;

final19_q7_data;

cvx_begin
    variables x(n) B0
    minimize(P'*x+B0)
    subject to
        -x <= 0
        -B0 <= 0
        B(1) = (1+rp)*B0
        for t=1:T-1
            B(t+1)= min((1+rp)*(B(t)+A(t,:)*x-E(t)),(1+rn)*(B(t)+A(t,:)*x-E(t)))
        end
        E(T)-B(T)-A(T,:)*x <= 0
cvx_end

% Compare to investment if no bonds were purchased

cvx_begin
    variable D0
    minimize(D0)
    subject to
        -D0 <= 0
        D(1) = (1+rp)*D0
        for t=1:T-1
            D(t+1)=min((1+rp)*(D(t)-E(t)),(1+rn)*(D(t)-E(t)));
        end
        E(T)-D(T) <= 0
cvx_end

disp("Optimal values of x:")
disp(x)
disp("Optimal value of B0:")
disp(B0)
disp("Optimal total initial investment:")
disp(P'*x+B0)
disp("Optimal investment without bonds:")
disp(D0)

figure(1)
plot(1:T,B)
title('Optimal Cash Balance with Investments')
xlabel('Period')
ylabel('Cash Balance')

figure(2)
plot(1:T,D)
title('Optimal Cash Balance without Investments')
xlabel('Period')
ylabel('Cash Balance')