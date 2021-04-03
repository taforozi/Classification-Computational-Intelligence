%% Fuzzy Systems - Classification (Part 1)
% Aforozi Thomais
% AEM 9291
% TSK Model 4 - Sub. Clustering (Class dependent)
% small radius - more rules

function [fis,OA_4,PA_4,UA_4,k_4,ErrorMatrix] = TSK_Model_4(training_data,validation_data,check_data,haberman)

% Subtractive Clustering per class (Class Dependent)
radius = 0.3;
% since we have two classes we use subtractive clustering for each class
% c1: centers of clusters, sig1: amplitude of gaussians (class 1)
% c2: centers of clusters, sig2: amplitude of gaussians (class 2)
[c1,sig1] = subclust(training_data(training_data(:,end)== 1,:),radius);
[c2,sig2] = subclust(training_data(training_data(:,end)== 2,:),radius);
num_rules = size(c1,1) + size(c2,1);

% Build FIS From Scratch
fis = newfis('FIS_SC','sugeno');

% Add Input-Output Variables
names_in = {'in1','in2','in3'};
for i=1:size(training_data,2)-1
    fis = addvar(fis,'input',names_in{i},[0 1]);
end
fis = addvar(fis,'output','out1',[1 2]);

% Add Input Membership Functions
name = 'sth';
for i = 1:size(training_data,2)-1
    for j = 1:size(c1,1)
        fis = addmf(fis,'input',i,name,'gaussmf',[sig1(i) c1(j,i)]);
    end
    for j = 1:size(c2,1)
        fis = addmf(fis,'input',i,name,'gaussmf',[sig2(i) c2(j,i)]);
    end
end

% Add Output Membership Functions
% for each cluster we add a constant value that we want as a return
params = [ones(1,size(c1,1)) 2*ones(1,size(c2,1))];
for i=1:num_rules
    fis = addmf(fis,'output',1,name,'constant',params(i));
end

% Add FIS Rule Base
ruleList = zeros(num_rules,size(training_data,2));
for i=1:size(ruleList,1)
    ruleList(i,:)=i;
end
ruleList = [ruleList ones(num_rules,2)];
fis = addrule(fis,ruleList);

% Train & Evaluate ANFIS
options = anfisOptions('InitialFis',fis,'ValidationData',validation_data,...
                        'EpochNumber',100);
% use validation data to avoid overfitting
[trnFis,trnError,~,valFis,valError] = anfis(training_data,options);

%% Evaluate TSK Model
output = round(evalfis(check_data(:,1:end-1),valFis));

%% Desired Metrics
% Error matrix
classes = unique(haberman(:, end));
dim = length(classes);
ErrorMatrix = zeros(dim);
N = length(check_data);
for i = 1:N
    xpos = find(classes == output(i));
    ypos = find(classes == haberman(i, end));
    ErrorMatrix(xpos, ypos) = ErrorMatrix(xpos, ypos) + 1;
end
 
% Overall accurancy (OA)
OA_4 = trace(ErrorMatrix) / N;
% or sum(diag(ErrorMatrix))/length(check_data(:,end));

% Producer's accurancy - User's accurancy (UA)
% N = sum(ErrorMatrix(:));
    
x_ir = sum(ErrorMatrix, 2); % sum of each row
x_jc = sum(ErrorMatrix, 1); % sum of each column

PA_4 = zeros(1, dim);
UA_4 = zeros(1, dim);
     
for i = 1:dim
PA_4(i) = ErrorMatrix(i,i) / x_jc(i);
UA_4(i) = ErrorMatrix(i,i) / x_ir(i);
end 

% hat{K}   
k_4 = (N * trace(ErrorMatrix) - PA_4 * UA_4') / (N^2 - PA_4 * UA_4');
 
%% Plots
% Membership functions
for i = 1:length(fis.input)
    figure;
    [xmf, ymf] = plotmf(fis, 'input', i);
    plot(xmf, ymf,'LineWidth',0.8);
    xlabel('Model 4 - Input (initial)', 'Interpreter', 'Latex');
    ylabel('Degree of membership', 'Interpreter', 'Latex');
    title(['Input #' num2str(i)]);
end
for i = 1:length(trnFis.input)
    figure;
    [xmf, ymf] = plotmf(trnFis, 'input', i);
    plot(xmf, ymf,'LineWidth',0.8);
    xlabel('Model 4 - Input (trained)', 'Interpreter', 'Latex');
    ylabel('Degree of membership', 'Interpreter', 'Latex');
    title(['Input #' num2str(i)]);
end

% learning curves 
figure;
plot(trnError, 'LineWidth', 1.5);
hold on;
plot(valError, 'LineWidth', 1.5);
title('Learning curves for TSK model 4','Interpreter','Latex'); 
grid on;
xlabel('Number of Iterations','Interpreter','Latex'); 
ylabel('Error','Interpreter','Latex');
leg1 = legend('Training Error','Validation Error');
set(leg1,'Interpreter','latex');
end
