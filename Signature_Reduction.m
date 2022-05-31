clc;
clear all;


%Initializing the parameter values   

I=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]; %current
B_x=zeros(1,29); 
B_y=zeros(1,29);
B_z=zeros(1,29);
u0=1.26*10^(-6);
u=700;
dl=2; %element length
N=10; % Number of DG coils
P=30; %Nunber of sample points
x= [0,2,4,6,8,10];
z= [0,-0.5, -0.49, 0.49, 0.5];
y=0;
sample_point = 1:1:29;
points= 1:1:30;
alphaA = [90,-90,14.04,-14.04,7.13,-7.13,6.31,-6.31,3.58,-3.58, 2.86,-2.86, 0,180,0,0,0,90, -90,13.77,-13.77,6.98,-6.98,5.36,-5.36,3.50,-3.50,2.81,-2.81];  
gammaA = [0,180,75.96,-75.96,82.87,-82.87,85.23,-85.23,86.42,-86.42,87.14,-87.14,90,-90,90,90,0,0,180,76.23,-76.23,83.01,-83.01,85.33,-85.33,86.49,-86.49,87.19,-87.19];
betaA  = [89.99,89.01,89.97, 89.86,89.54,89.01, 88.34,90.01,90.05,89.98, 89.99, 90, 90.03, 89.97, 89.95, 88.99, 90, 90.07, 90.01, 89.92, 89.99,90.01, 90.03, 89.99, 89.95, 90, 90.04, 89.92, 89.99]; 
B_x=zeros(1,29);
B_y=zeros(1,29);
B_z=zeros(1,29);
R=zeros(1,P);

%Calculating the distance matrix R for all sample points
for i=1:1:length(x)
    for j=1:1:length(z)
     
        R(i,j) = sqrt(x(i)^2+z(j)^2);
        
    end
end 
Rnew= nonzeros(R);
Rfinal =  Rnew';



%Importing data from excel file
[data_without_dg, strings, raw] = xlsread("Predicted Signatures.xls");
sample_point = 1:1:29;
points= 1:1:30;

B_x_without_dg = data_without_dg(1:30);
B_y_without_dg = data_without_dg(31:60);
B_z_without_dg = data_without_dg(61:90);

%Reinforcement Learning for X Component
state=B_x_without_dg*10^(11);
action=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]; 

% initial Q matrix
gamma = 0.5;    % discount factor
alpha = 0.5;    % learning rate   
epsilon = 0.9;  % exploration probability (1-epsilon = exploit and epsilon = explore)
Q = zeros(length(state),length(action));
K = 1000;     % maximum number of the iterations
state_idx = 15;  % the initial state to begin from
%The main loop of algorithm starts here
for k = 1:K
    disp(['iteration: ' num2str(k)]);
    r=rand; % get 1 uniform random number
   
    % choose either explore or exploit
    if state == 0   % exploit
        [~,umax]=max(Q(state_idx,:));
        current_action = action(umax);
    else        % explore
        current_action=datasample(action,1); % choose 1 action randomly (uniform random distribution)
    end
    
    action_idx = find(action==current_action); % id of the chosen action
   
    if (state_idx==29)
        state_idx=1;
    else    
    u=state(state_idx+1)-state(state_idx);
    end
    [next_state,next_reward] = model(state(state_idx),u);
  
    next_state_idx=state_idx+1;
    % print the results in each iteration
    disp(['current state : ' num2str(state(state_idx)) ' next state : ' num2str(state(next_state_idx)) ' taken action : ' num2str(action(action_idx))]);
    disp([' next reward : ' num2str(next_reward)]);
    % update the Q matrix using the Q-learning rule
    Q(state_idx,action_idx) = Q(state_idx,action_idx) + alpha * (next_reward + gamma* max(Q(next_state_idx,:)) - Q(state_idx,action_idx));

    if (next_state_idx == 0 || next_state_idx == 30)
        state_idx = datasample(2:length(state)-1,1); % we just restart the episode with a new state
    else
        state_idx = next_state_idx;
    end
    disp(Q);  % display Q in each level
end
% display the final Q matrix
disp('Final Q matrix : ');
disp(Q)
[C,I]=max(Q,[],2);                              % finding the max values
disp('Q(optimal):');
disp(C);
disp("I =")
disp(I)
disp('Optimal Policy');
disp('*');
disp([action(I(2,1));action(I(3,1));action(I(4,1));action(I(5,1));action(I(5,1));action(I(6,1));action(I(7,1));action(I(8,1));action(I(9,1));action(I(10,1))]);
disp('*');


%After Reinforcement Learning
I=[action(I(2,1));action(I(3,1));action(I(4,1));action(I(5,1));action(I(5,1));action(I(6,1));action(I(7,1));action(I(8,1));action(I(9,1));action(I(10,1))];
I=I';

%Calculating the total magnetic flux density  
B_new=zeros(N,length(Rfinal));
B_x_new=zeros(1,29);
B_y_new=zeros(1,29);
B_z_new=zeros(1,29);
for i = 1:N
     for j= 1:length(Rfinal)
           B_new(i,j)=(I(i)*u0/(u*4*pi*(Rfinal(1,j)^2)))*dl*sind(betaA(j));
     end
end
B_new_Total=sum(B_new);

%Calculating the three magnetic flux density components
for i = 1: length(alphaA)
  B_x_new(1,i)= B_new_Total(1,i)*cosd(alphaA(i));
end
for i = 1: length(gammaA) 
  B_z_new(1,i)= B_new_Total(1,i)*cosd(gammaA(i));
end
for i = 1: length(betaA)
  B_y_new(1,i)= B_new_Total(1,i)*cosd(betaA(i));
end


figure(1)
plot(points, B_x_without_dg);
hold on;
plot(sample_point, B_x_new,'-r','LineWidth',2);
legend('X Signatures', 'DG with RL Tuning');
xlabel("Sample points(m)");
ylabel("Magnetic Signatures X Component(T)");

figure(2)
plot(points, B_y_without_dg);
hold on;
plot(sample_point, B_y_new,'-r','LineWidth',2);
legend('Y signatures', 'DG with RL Tuning');
xlabel("Sample points(m)");
ylabel("Magnetic Signatures X Component(T)");

figure(3)
plot(points, B_z_without_dg);
hold on;
plot(sample_point, B_z_new,'-r','LineWidth',2);
legend('Z Signatures', 'DG with RL Tuning');
xlabel("Sample points(m)");
ylabel("Magnetic Signatures X Component(T)");

B_x_ref=zeros(1,29);
B_y_ref=zeros(1,29);
B_z_ref=zeros(1,29);
B_x_diff=zeros(1,29);
B_y_diff=zeros(1,29);
B_z_diff=zeros(1,29);
for i=1:29
    B_x_diff(1,i)= abs(B_x_new(1,i)-B_x_ref(1,i));
    B_y_diff(1,i)= abs(B_y_new(1,i)-B_y_ref(1,i));
    B_z_diff(1,i)= abs(B_z_new(1,i)-B_z_ref(1,i));
end

% figure(4)
% plot(sample_point, B_x_diff*10^(-1),'b', 'Linewidth', 4);
% %title("Error for x component for 1000 iterations");
% xlabel("Sample points(m)");
% ylabel("Difference between signatures(T)");
% legend('10 iterations', '100 iterations', '1000 iterations');
% 
% figure(5)
% plot(sample_point, B_y_diff, 'Linewidth', 4);
% %title("Error for y component for 1000 iterations");
% xlabel("Sample points(m)");
% ylabel("Difference between signatures(T)");
% legend('10 iterations', '100 iterations', '1000 iterations');
% 
% 
% figure(6)
% plot(sample_point, B_z_diff, 'Linewidth', 4);
% %title("Error for z component for 1000 iterations");
% xlabel("Sample points(m)");
% ylabel("Difference between signatures(T)");
% legend('10 iterations', '100 iterations', '1000 iterations');







% Function to give next state and reward using current state and action
function [next_state,r] = model(state,u)
if (state < 30 && state>=0)
     next_state = state + u;
 else
     next_state = state;
end
if (state<=5)
    r = 100;
elseif (state>5 &&state<=10)
    r = 75;
    
elseif (state>10 &&state<=15)
    r = 50;
elseif (state>15 &&state<=20)
    r = 25;
elseif (state>20 &&state<=25)
    r=10;
 
else
    r = 0;
end
end










     
    
        




