%% Fuzzy Systems - Classification (Part 2)
% Aforozi Thomais
% AEM 9291
% Grid Search - TSK Model - Sub. Clustering (Class dependent)

%% clear workspace
% clear all;
% close all;

%% Load & Prepare the data
fprintf('Load & prepare dataset... \n');
dataset = csvread('data.csv',1,1);

[training_data, validation_data, check_data] = split_scale(dataset,1);

%% Trial features & radius
NF = [3 6 9 12]; 
radii = [0.3 0.6 0.9];

fprintf('Select features (relieff)... \n');
% select features
[ranks,weights] = relieff(training_data(:,1:end-1),training_data(:,end), 10);

%% Grid Search & 5-fold Cross Validation
meanError = zeros(length(NF),length(radii));
fprintf('Loop... \n');
pointer =1;

observations = length(training_data(:,end));

for f = 1:length(NF)
    for r = 1:length(radii)
        
    % 5-fold Cross Validation
    cv = cvpartition(observations, 'KFold', 5);
     
    for k = 1:5
        fprintf(['Repetition: ', num2str(pointer)]);
        
        % select features
        features_selected = [training_data(:,ranks(1:NF(f))) training_data(:,end)];
        
        % Getting the indices of the randomly splitted data with 5-fold cross
        % validation
        vali = test(cv,k); 
        train = training(cv,k);
            
        valIdx = find(vali == 1);
        trainIdx = find(train == 1);
          
        training_data_new = features_selected(trainIdx,:);
        validation_data_new = features_selected(valIdx,:);
          
        % since we have five classes we use subtractive clustering for each class
        [c1,sig1] = subclust(training_data_new(training_data_new(:,end)== 1,:),radii(r));
        [c2,sig2] = subclust(training_data_new(training_data_new(:,end)== 2,:),radii(r));
        [c3,sig3] = subclust(training_data_new(training_data_new(:,end)== 3,:),radii(r));
        [c4,sig4] = subclust(training_data_new(training_data_new(:,end)== 4,:),radii(r));
        [c5,sig5] = subclust(training_data_new(training_data_new(:,end)== 5,:),radii(r));
        num_rules = size(c1,1) + size(c2,1) + size(c3,1) + size(c4,1) + size(c5,1);

        % Build FIS From Scratch
        fis = newfis('FIS_SC','sugeno');

        % Add Input-Output Variables
        names_in = cell(size(training_data_new,2)-1,1);
        for i = 1:size(training_data_new,2)-1
            names_in{i} = sprintf("in%d",i);
        end
        for i=1:size(training_data_new,2)-1
            fis = addvar(fis,'input',names_in{i},[0 1]);
            % eisodos [0 1] me tin kanonikopoihsh
        end
        fis = addvar(fis,'output','out1',[1 5]);

        % Add Input Membership Functions
        name = 'sth';
        for i=1:size(training_data_new,2)-1
            for j=1:size(c1,1)
                fis = addmf(fis,'input',i,name,'gaussmf',[sig1(i) c1(j,i)]);
            end
            for j=1:size(c2,1)
                fis = addmf(fis,'input',i,name,'gaussmf',[sig2(i) c2(j,i)]);
            end
            for j=1:size(c3,1)
                fis = addmf(fis,'input',i,name,'gaussmf',[sig3(i) c3(j,i)]);
            end
            for j=1:size(c4,1)
                fis = addmf(fis,'input',i,name,'gaussmf',[sig4(i) c4(j,i)]);
            end
            for j=1:size(c5,1)
                fis = addmf(fis,'input',i,name,'gaussmf',[sig5(i) c5(j,i)]);
            end
        end

        % Add Output Membership Functions
        % for each cluster we add a constant value that we want as a return
        params = [ones(1,size(c1,1)) 2*ones(1,size(c2,1)) 3*ones(1,size(c3,1)) ...
                    4*ones(1,size(c4,1)) 5*ones(1,size(c5,1))];
        for i = 1:num_rules
            fis = addmf(fis,'output',1,name,'constant',params(i));
        end

        % Add FIS Rule Base
        ruleList = zeros(num_rules,size(training_data_new,2));
        for i=1:size(ruleList,1)
            ruleList(i,:)=i;
        end
        ruleList = [ruleList ones(num_rules,2)];
        fis = addrule(fis,ruleList);

        % Train & Evaluate ANFIS
%         options = anfisOptions('InitialFis',fis,'ValidationData',validation_data_new,...
%                                 'EpochNumber',100);
        % use validation data to avoid overfitting
        [trnFis,trnError,~,valFis,valError] = anfis(training_data_new,fis,[100 0 0.01 0.9 1.1],[],validation_data_new);
        
%         anfis(training_data_new,options);
        meanError(f,r) =  meanError(f,r) + mean(valError);
        pointer = pointer + 1;
    end
    end    
end

meanError = meanError / 5;

%% Find the optimal NF & radius combination
fprintf('Find optimal values... \n');
[optimal_NF, optimal_radius] = find(meanError == min(meanError(:)));

des_features = NF(optimal_NF);
desired_rad = radii(optimal_radius);
fprintf(['Optimal NF: ', num2str(des_features)]);
fprintf(['\n Optimal radius: ', num2str(desired_rad)]);

optimal_data = dataset(:,ranks(1:des_features));
opt_training_data = [training_data(:,ranks(1:des_features)) training_data(:,end)];
opt_validation_data = [validation_data(:,ranks(1:des_features)) validation_data(:,end)];
opt_check_data = [check_data(:,ranks(1:des_features)) check_data(:,end)];

%% Train the optimal TSK model
% since we have five classes we use subtractive clustering for each class
[final_c1,final_sig1] = subclust(opt_training_data(opt_training_data(:,end)== 1,:),desired_rad);
[final_c2,final_sig2] = subclust(opt_training_data(opt_training_data(:,end)== 2,:),desired_rad);
[final_c3,final_sig3] = subclust(opt_training_data(opt_training_data(:,end)== 3,:),desired_rad);
[final_c4,final_sig4] = subclust(opt_training_data(opt_training_data(:,end)== 4,:),desired_rad);
[final_c5,final_sig5] = subclust(opt_training_data(opt_training_data(:,end)== 5,:),desired_rad);
final_num_rules = size(final_c1,1) + size(final_c2,1) + size(final_c3,1) + size(final_c4,1) + size(final_c5,1);

 % Build FIS From Scratch
optimal_fis = newfis('FIS_SC','sugeno');

 % Add Input-Output Variables
names_in2 = cell(size(training_data_new,2)-1,1);
for i = 1:size(training_data_new,2)-1
    names_in2{i} = sprintf("in%d",i);
end
for i=1:size(opt_training_data,2)-1
    optimal_fis= addvar(optimal_fis,'input', names_in2{i},[0 1]);
end
optimal_fis = addvar(optimal_fis,'output','out1',[1 5]);

% Add Input Membership Functions
name = 'sth';
for i=1:size(opt_training_data,2)-1
    for j=1:size(final_c1,1)
        optimal_fis = addmf(optimal_fis,'input',i,name,'gaussmf',[final_sig1(i) final_c1(j,i)]);
    end
    for j=1:size(final_c2,1)
        optimal_fis = addmf(optimal_fis,'input',i,name,'gaussmf',[final_sig2(i) final_c2(j,i)]);
    end
    for j=1:size(final_c3,1)
        optimal_fis = addmf(optimal_fis,'input',i,name,'gaussmf',[final_sig3(i) final_c3(j,i)]);
    end
    for j=1:size(final_c4,1)
        optimal_fis = addmf(optimal_fis,'input',i,name,'gaussmf',[final_sig4(i) final_c4(j,i)]);
    end
    for j=1:size(final_c5,1)
        optimal_fis = addmf(optimal_fis,'input',i,name,'gaussmf',[final_sig5(i) final_c5(j,i)]);
    end
end

% Add Output Membership Functions
% for each cluster we add a constant value that we want as a return
final_params = [ones(1,size(final_c1,1)) 2*ones(1,size(final_c2,1)) 3*ones(1,size(final_c3,1)) ...
                4*ones(1,size(final_c4,1)) 5*ones(1,size(final_c5,1))];
for i = 1:final_num_rules
    optimal_fis = addmf(optimal_fis,'output',1,name,'constant',final_params(i));
end

% Add FIS Rule Base
final_ruleList = zeros(final_num_rules,size(opt_training_data,2));
for i=1:size(final_ruleList,1)
    final_ruleList(i,:)=i;
end
final_ruleList = [final_ruleList ones(final_num_rules,2)];
optimal_fis = addrule(optimal_fis,final_ruleList);

% Train & Evaluate ANFIS
% use validation data to avoid overfitting
options = anfisOptions('InitialFis',optimal_fis,'ValidationData',opt_validation_data,...
                       'EpochNumber',100);
[trnFis_opt,trnError_opt,~,valFis_opt,valError_opt] = anfis(opt_training_data,options);
%% Evaluate TSK Model
output = round(evalfis(opt_check_data(:,1:end-1),valFis_opt));

%% Desired Metrics
% Error matrix
classes = unique(dataset(:, end));
dim = length(classes);
ErrorMatrix = zeros(dim);
N = length(check_data);
for i = 1:N
    xpos = find(classes == output(i));
    ypos = find(classes == dataset(i, end));
    ErrorMatrix(xpos, ypos) = ErrorMatrix(xpos, ypos) + 1;
end
 
% Overall accurancy (OA)
 OA = trace(ErrorMatrix) / N;

% Producer's accurancy - User's accurancy (UA)  
x_ir = sum(ErrorMatrix, 2); % sum of each row
x_jc = sum(ErrorMatrix, 1); % sum of each column

PA = zeros(1, dim);
UA = zeros(1, dim);
     
for i = 1:dim
PA(i) = ErrorMatrix(i,i) / x_jc(i);
UA(i) = ErrorMatrix(i,i) / x_ir(i);
end 

% hat{K}   
k = (N * trace(ErrorMatrix) - PA * UA') / (N^2 - PA * UA');

%% Plots
% predictions and real values
figure;
hold on;
grid on;
title('Predictions and Real values','Interpreter','Latex');
xlabel('Test dataset sample','Interpreter','Latex');
ylabel('y','Interpreter','Latex');
plot(1:length(output), output,'o','Color','blue');
plot(1:length(output), opt_check_data(:, end),'o','Color','red');
leg = legend('Predictions', 'Real values');
set(leg,'Interpreter','latex');


% learning curves 
figure;
plot(trnError_opt, 'LineWidth', 1.5);
hold on;
plot(valError_opt, 'LineWidth', 1.5);
title('Learning curves for TSK model','Interpreter','Latex'); 
grid on;
xlabel('Number of Iterations','Interpreter','Latex'); 
ylabel('Error','Interpreter','Latex');
leg1 = legend('Training Error','Validation Error');
set(leg1,'Interpreter','latex');

% Membership functions
for i = 1:size(opt_training_data,2)-1 % 1:length(fis.input)
 
     % initial memberships
     figure;
     [xmf, ymf] = plotmf(fis, 'input', i);
     plot(xmf, ymf,'LineWidth',0.8);
     xlabel('Input (initial)', 'Interpreter', 'Latex');
     ylabel('Degree of membership', 'Interpreter', 'Latex');
     title(['Input #' num2str(i)]);
     
     figure;
     [xmf2, ymf2] = plotmf(trnFis_opt, 'input', i);
     plot(xmf2, ymf2,'LineWidth',0.8);
     xlabel('Input (trained)', 'Interpreter', 'Latex');
     ylabel('Degree of membership', 'Interpreter', 'Latex');
     title(['Input #' num2str(i)]); 
end


% Mean error respective to number of featuresfigure
figure;
hold on
grid on
title('Mean error respective to number of features','Interpreter','Latex');
xlabel('Number of features', 'Interpreter', 'Latex');
ylabel('Error', 'Interpreter', 'Latex');
plot(NF, meanError(:, 1), 'LineWidth', 1.5);
plot(NF, meanError(:, 2), 'LineWidth', 1.5);
plot(NF, meanError(:, 3), 'LineWidth', 1.5);
leg2 = legend('0.3 radius', '0.6 radius', '0.9 radius');
set(leg2,'Interpreter','Latex');

% Mean error respective to the cluster radius
figure
hold on
grid on
title('Mean error respective to the cluster radius','Interpreter','Latex');
xlabel('Cluster radius', 'Interpreter', 'Latex');
ylabel('Error', 'Interpreter', 'Latex');
plot(radii, meanError(1, :), 'LineWidth', 1.5);
plot(radii, meanError(2, :), 'LineWidth', 1.5);
plot(radii, meanError(3, :), 'LineWidth', 1.5);
plot(radii, meanError(4, :), 'LineWidth', 1.5);
% axis([0.3 0.95 1.02 1.2]);
leg3 = legend('3 features', '6 features', '9 features','12 features');
set(leg3,'Interpreter','Latex');


