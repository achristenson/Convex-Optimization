%% Question 4 - Take Home Final Exam - Alexander Christenson

clear all; close all;

final19_q4_data

C = 3;
D = 3;
Q = 35;

cvx_begin quiet
    variables c(T) q(T)
    minimize(p'*(u+c))
    subject to
        c <= C
        -c <= D
        q <= Q
        -q <=0
        for k=1:T-1
            c(k) == q(k+1) - q(k)
        end
        c(T) == q(1) - q(T)
cvx_end

plot(t/4,c) % I tried to match the formatting from the data file with this, so the time is scaled by a factor of 4
legend('price','usage','charge rate')
title('Plot of usage, price, and charge over time')
xlabel('time')
hold off

% Part c - plot the minimum total cost vs Q
Q_max = Q;
p_star = [];
for(Q=1:Q_max)
    cvx_begin quiet
        variables c1(T) q1(T)
        minimize(p'*(u+c1))
        subject to
            c1 <= C
            -c1 <= D
            q1 <= Q
            -q1 <=0
            for k=1:T-1
                c1(k) == q1(k+1) - q1(k)
            end
            c1(T) == q1(1) - q1(T)
    cvx_end
    
    p_star = [p_star;cvx_optval];

end

figure()
plot(1:Q_max,p_star)
hold on

C = 1;
D = 1;
p_star1 = [];

for(Q=1:Q_max)
    cvx_begin quiet
        variables c1(T) q1(T)
        minimize(p'*(u+c1))
        subject to
            c1 <= C
            -c1 <= D
            q1 <= Q
            -q1 <=0
            for k=1:T-1
                c1(k) == q1(k+1) - q1(k)
            end
            c1(T) == q1(1) - q1(T)
    cvx_end
    
    p_star1 = [p_star1;cvx_optval];

end

plot(1:Q_max,p_star1)
legend('C=D=3','C=D=1')
title("Trade off between minimum total cost and capacity")
xlabel("Capacity")
ylabel("Minimum total cost")

%{
What the endpoints of the graph tell us is that the capacity and discharge
rates are very important in determining the value of the battery. The
general trend is that as the capacity increases the total cost is lowered,
however this is limited by how quickly the battery can charge/discharge. It
also tells us that faster batteries lower the cost more.
%}
