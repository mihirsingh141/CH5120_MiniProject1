clc;
clear;
close all;


%Initial Values

A1 = 28; %(cm^2)
A2 = 32;
A3 = 28;
A4 = 32;

a1 = 0.071; a3 = 0.071; %(cm^2)
a2 = 0.057; a4 = 0.057;

kc = 0.5; %(V/cm)
g = 981; %(cm/s^2)

gamma1 = 0.7; gamma2 = 0.6; %constants that are determined from valve postion

k1 = 3.33; k2 = 3.35; %(cm^3/Vs)

v1 = 3;v2 = 3; %(V)

h0 = [12.4; 12.7; 1.8; 1.4];


C = [kc, 0, 0, 0; 0, kc, 0, 0];


Q = 0.00001*eye(4); %covariance matrix of noise in state equation
R = 0.00001*eye(2); %covariance matrix of noise in measurement equation

%Ordinary Differential Equation System

tspan = linspace(0,6000,10000);
    
[t,h] = ode45(@ODE,tspan,h0);
y = C*h' + R*randn(2,10000);


%Visualization of Simulation Results


figure;
plot(t,h(:,1),t,h(:,2),t,h(:,3),t,h(:,4))
legend('h1','h2','h3','h4')
title('Simulation results')

%Initialization

x0 = h0;
Ts = tspan(2)-tspan(1);

n = 4; %Number of states
P0 =  0.0001*eye(4);%Initial Covariance Matrix

N = 400; %Number of particles

L = chol(P0); 
x0 = (x0*ones(1,N)) + L*rand(n,N); %Adding covariance to each particle


x_post = x0;

%Roughing the posterior with Noise
w = chol(Q)*rand(n,N);
w1 = w(1,:);
w2 = w(2,:);
w3 = w(3,:);
w4 = w(4,:);

x_post(1,:) = x_post(1,:) + w1;
x_post(2,:) = x_post(2,:) + w2;
x_post(3,:) = x_post(3,:) + w3;
x_post(4,:) = x_post(4,:) + w4;

x_prior_mean = zeros(4,10001);
x_post_mean = zeros(4,10001);


k=1;
while(k<=10000)
    w = chol(Q)*randn(n,N); %for roughening the prior

    x_dt = zeros(4,N);

    %Prediction 
    
     x_dt(1,:) = (-a1/A1*sqrt(2*g*(x_post(1,:))) + a3/A1*sqrt(2*g*(x_post(3,:))) + (gamma1*k1*v1)/A1);
     x_dt(2,:) = -a2/A2*sqrt(2*g*(x_post (2,:))) + a4/A2*sqrt (2*g*(x_post(4,:))) + (gamma2*k1*v2)/A2 ;
     x_dt(3,:) = -a3/A3*sqrt (2*g*(x_post(3,:))) + (1 - gamma2)*k2*v2/A3;
     x_dt(4,:) = -a4/A4*sqrt (2*g*(x_post(4,:))) + (1 - gamma1)*k1*v1/A4;

     x_prior = x_post + x_dt*Ts + w;

     

     %Importance weights
     z_true = y(:,k) * ones(1,N);
     z_est = C*x_prior;

     v = z_true-z_est;

     q = zeros(1,N);
     wt = zeros(1,N);

     for i=1:N
         q(i) = exp(-0.5 * v(:,i)'*inv(R)*v(:,i));
     end

     for i=1:N
         wt(i) = q(i)/sum(q);
     end
       
     j=1;
     cumsum_wt = cumsum(wt);
     for i=1:N
         r = rand;
         while cumsum_wt(j) < r
             j=j+1;
         end
         x_post(:,i) = x_prior(:,j);
     end
     

     x_prior_mean(:,k) = mean(x_prior,2);
     x_post_mean(:,k) = mean(x_post,2);

     k=k+1;

end

%Visualization

k_span = 1:10000;

figure;
plot(k_span,x_post_mean(1,1:10000),k_span,x_post_mean(2,1:10000),k_span,x_post_mean(3,1:10000),k_span,x_post_mean(4,1:10000));
legend('h1','h2','h3','h4')
title('Particle Filter results')

estm_error = h' - x_post_mean(:,1:10000);

figure;
plot(k_span,estm_error(1,1:10000),k_span,estm_error(2,1:10000),k_span,estm_error(3,1:10000),k_span,estm_error(4,1:10000));
legend('h1','h2','h3','h4')
title('Residuals plot')

figure;
subplot(2,2,1)
plot(k_span,h(:,1)',k_span,x_post_mean(1,1:10000))
legend('Simulated','estimated')
title('h1');

subplot(2,2,2)
plot(k_span,h(:,2)',k_span,x_post_mean(2,1:10000))
legend('Simulated','estimated')
title('h2');

subplot(2,2,3)
plot(k_span,h(:,3)',k_span,x_post_mean(3,1:10000))
legend('Simulated','estimated')
title('h3');

subplot(2,2,4)
plot(k_span,h(:,4)',k_span,x_post_mean(4,1:10000))
legend('Simulated','estimated')
title('h4');

sgtitle('Comparing the estimates with the actual values')


function dhdt = ODE(t,h)
    A1 = 28; %(cm^2)
    A2 = 32;
    A3 = 28;
    A4 = 32;
    
    a1 = 0.071; a3 = 0.071; %(cm^2)
    a2 = 0.057; a4 = 0.057;
    
    kc = 0.5; %(V/cm)
    g = 981; %(cm/s^2)
    
    gamma1 = 0.7; gamma2 = 0.6; %constants that are determined from valve postion
    
    k1 = 3.33; k2 = 3.35; %(cm^3/Vs)
    
    v1 = 3;v2 = 3; %(V)

    dhdt = zeros(4,1);

    dhdt(1) = -a1/A1*(2*g*h(1))^0.5 + a3/A1*(2*g*h(3))^0.5 + gamma1*k1/A1*v1;
    dhdt(2) = -a2/A2*(2*g*h(2))^0.5 + a4/A2*(2*g*h(4))^0.5 + gamma2*k2/A2*v2;
    dhdt(3) = -a3/A3*(2*g*h(3))^0.5 + (1-gamma2)*k2*v2/A3;
    dhdt(4) = -a4/A4*(2*g*h(4))^0.5 + (1-gamma1)*k1*v1/A4;
end