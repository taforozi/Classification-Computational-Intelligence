%% Fuzzy Systems - Classification (Part 1)
% Aforozi Thomais
% AEM 9291
% TSK Model 1 - Sub. Clustering (Class independed)
% large radius - less rules

function [fis,OA_1,PA_1,UA_1,k_1,ErrorMatrix] = TSK_Model_1(training_data,validation_data,check_data,haberman)

% Subtractive Clustering (Class Independed)
fis = genfis2(training_data(:,1:end-1),training_data(:,end),0.8);

for i = 1:length(fis.output.mf)
    fis.output.mf(i).type = 'constant';
    fis.output.mf(i).params = fis.output.mf(i).params(end); 
end

% Train TSK Model
% use validation data to avoid overfitting
options = anfisOptions('InitialFis',fis,'ValidationData',validation_data,...
                        'EpochNumber',100);
[trnFis,trnError,~,valFis,valError] = anfis(training_data,options);

% Evaluate TSK Model
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
OA_1 = trace(ErrorMatrix) / N;

% Producer's accurancy - User's accurancy (UA)  
x_ir = sum(ErrorMatrix, 2); % sum of each row
x_jc = sum(ErrorMatrix, 1); % sum of each column

PA_1 = zeros(1, dim);
UA_1 = zeros(1, dim);
     
for i = 1:dim
PA_1(i) = ErrorMatrix(i,i) / x_jc(i);
UA_1(i) = ErrorMatrix(i,i) / x_ir(i);
end 

% hat{K}   
k_1 = (N * trace(ErrorMatrix) - PA_1 * UA_1') / (N^2 - PA_1 * UA_1');
 
%% Plots
% Membership functions
for i = 1:length(fis.input)
    figure;
    [xmf, ymf] = plotmf(fis, 'input', i);
    plot(xmf, ymf,'LineWidth',0.8);
    xlabel('Model 1 - Input (initial)', 'Interpreter', 'Latex');
    ylabel('Degree of membership', 'Interpreter', 'Latex');
    title(['Input #' num2str(i)]);
end
for i = 1:length(trnFis.input)
    figure;
    [xmf, ymf] = plotmf(trnFis, 'input', i);
    plot(xmf, ymf,'LineWidth',0.8);
    xlabel('Model 1 - Input (trained)', 'Interpreter', 'Latex');
    ylabel('Degree of membership', 'Interpreter', 'Latex');
    title(['Input #' num2str(i)]);
end

% learning curves 
figure;
plot(trnError, 'LineWidth', 1.5);
hold on;
plot(valError, 'LineWidth', 1.5);
title('Learning curves for TSK model 1','Interpreter','Latex'); 
grid on;
xlabel('Number of Iterations','Interpreter','Latex'); 
ylabel('Error','Interpreter','Latex');
leg1 = legend('Training Error','Validation Error');
set(leg1,'Interpreter','latex');
end
