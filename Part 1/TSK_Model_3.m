%% Fuzzy Systems - Classification (Part 1)
% Aforozi Thomais
% AEM 9291
% TSK Model 3 - Sub. Clustering (Class dependent)
% large radius - less rules

function [fis,OA_3,PA_3,UA_3,k_3,ErrorMatrix] = TSK_Model_3(training_data,validation_data,check_data,haberman)

% Subtractive Clustering per class (Class Dependent)
radius = 0.8;
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
for i=1:size(training_data,2)-1
    for j=1:size(c1,1)
        fis = addmf(fis,'input',i,name,'gaussmf',[sig1(i) c1(j,i)]);
    end
    for j=1:size(c2,1)
        fis = addmf(fis,'input',i,name,'gaussmf',[sig2(i) c2(j,i)]);
    end
end

% Add Output Membership Functions
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

%% Desired Metrics% Error matrix
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
OA_3 = trace(ErrorMatrix) / N;
% or sum(diag(ErrorMatrix))/length(check_data(:,end));

% Producer's accurancy - User's accurancy (UA)
% N = sum(ErrorMatrix(:));
    
x_ir = sum(ErrorMatrix, 2); % sum of each row
x_jc = sum(ErrorMatrix, 1); % sum of each column

PA_3 = zeros(1, dim);
UA_3 = zeros(1, dim);
     
for i = 1:dim
PA_3(i) = ErrorMatrix(i,i) / x_jc(i);
UA_3(i) = ErrorMatrix(i,i) / x_ir(i);
end 

% hat{K}   
k_3 = (N * trace(ErrorMatrix) - PA_3 * UA_3') / (N^2 - PA_3 * UA_3');
 
%% Plots
% Membership functions
% Membership functions
for i = 1:length(fis.input)
    figure;
    [xmf, ymf] = plotmf(fis, 'input', i);
    plot(xmf, ymf,'LineWidth',0.8);
    xlabel('Model 3 - Input (initial)', 'Interpreter', 'Latex');
    ylabel('Degree of membership', 'Interpreter', 'Latex');
    title(['Input #' num2str(i)]);
end
for i = 1:length(trnFis.input)
    figure;
    [xmf, ymf] = plotmf(trnFis, 'input', i);
    plot(xmf, ymf,'LineWidth',0.8);
    xlabel('Model 3 - Input (trained)', 'Interpreter', 'Latex');
    ylabel('Degree of membership', 'Interpreter', 'Latex');
    title(['Input #' num2str(i)]);
end

% learning curves 
figure;
plot(trnError, 'LineWidth', 1.5);
hold on;
plot(valError, 'LineWidth', 1.5);
title('Learning curves for TSK model 3','Interpreter','Latex'); 
grid on;
xlabel('Number of Iterations','Interpreter','Latex'); 
ylabel('Error','Interpreter','Latex');
leg1 = legend('Training Error','Validation Error');
set(leg1,'Interpreter','latex');
end
