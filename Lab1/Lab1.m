%This Lab uses a Single-Neuron Perceptron Network to classify input 2D
%input vectors in one of two classifications.

%Define Training Dataset
N = 6;
p = [1 2 3 1 2 4; 4 5 3.5 0.5 2 0.5];
t = [1 1 1 0 0 0];

%Initialize All Flags to 1
flags = ones(1,N);

%Initialize Weights and Bias to zeros
W = [0 0];
b = 0;

%Loop while at least one input vector is still being misclassified
while atLeastOneFlagNotZero(flags, N) == 1
    
    %Reset flags to ones
    flags = ones(1, N);
    
    %Loop through each input vector
    for k = 1:N
        p_k = p(:,k);               %Input vector k
        a_k = hardlim(W*p_k + b);   %Calculated network output for input vector k
        e_k = t(k) - a_k;           %Calculated perceptron error for input vector k
        
        %Set the corresponding flag to zero if the input vector was
        %classified correctly. If the classification was incorrect, update
        %the weights and bias
        if e_k == 0
           flags(k) = 0; 
        else
            W(1) = W(1) + e_k * p_k(1);
            W(2) = W(2) + e_k * p_k(2);
            b = b + e_k; 
        end
    end
end

%Define the range of our x-axis (p1 axis)
x_begin = -1;
x_end = 6;
x = linspace(x_begin, x_end, x_end - x_begin);

%Define the decision boundary. 
%Solved for y from W(1)*x+W(2)*y+b = 0
r = (-W(1))/(W(2));
q = (-b)/(W(2));
y = r*x + q;

%Determine midpoint to draw the weight vector from
midPointX = (x_begin + x_end) / 2;
midPointY = r*midPointX + q;

%Create a unit vector in the direction of the W vector
lengthW = sqrt(W(1)^2 + W(2)^2);
W_direction_X_component = W(1) / lengthW;
W_direction_Y_component = W(2) / lengthW;

%Setup the figure window for the plot
x0=300;
y0=100;
width=700;
height=575;
set(gcf,'position',[x0,y0,width,height]);

%Plot the decision boundary
plot(x, y); xlabel('p1'); ylabel('p2'); title('Lab 1 Single-Neuron Perceptron Network');
hold on;

%Show the unit weight direction vector on the plot
text(midPointX - 0.3, midPointY + 0.6, "W");
quiver(midPointX, midPointY, W_direction_X_component, W_direction_Y_component, 0);

%Show the Class 1 Points on the plot 
for i=1:3
   p_i = p(:,i);
   scatter(p_i(1), p_i(2), 'filled', 'black');
end

%Show the Class 2 Points on the plot 
for i=4:6
   p_i = p(:,i);
   scatter(p_i(1), p_i(2), 'filled', 'red');
end

%After training the network, ask user for new input vectors to classify
xToClassify = input("Please enter the p1 element of the new input vector: ");
yToClassify = input ("Please enter the p2 element of the new input vector: ");
while (xToClassify >= -1) & (yToClassify >= 0)
    result = classifyInput(xToClassify, yToClassify, r, q);
    if result == 1
        fprintf("[%d %d] is classified as class 1\n", xToClassify, yToClassify);
    elseif result == 0
        fprintf("[%d %d] is classified as class 0\n", xToClassify, yToClassify);
    elseif result == -1
        fprintf("[%d %d] lies on the decision boundary and can't be classified.\n", xToClassify, yToClassify);
    end
    xToClassify = input("Please enter the p1 element of the new input vector: ");
    yToClassify = input ("Please enter the p2 element of the new input vector: ");
end
 fprintf("Exiting\n");

%Function to check if any of the flags are still non-zero
function [result] = atLeastOneFlagNotZero(flags, N)
    result = 0;
    for i = 1:N
       if flags(i) ~= 0
           result = 1;
       end
    end
end

function [result] = classifyInput(x, y, r, q)
    expectedY = r*x + q;
    if y > expectedY
        result = 1; % Class 1
        scatter(x, y, 'filled', 'blue');
    elseif y == expectedY
        result = -1; %On decision boundary, can't decide.
        scatter(x, y);
    else
        result = 0; % Class 2
        scatter(x, y, 'filled', 'blue');
    end
end
