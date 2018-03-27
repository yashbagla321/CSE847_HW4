%% Setup parameters
data_file = fullfile('spam_email','data.txt');
labels_file = fullfile('spam_email','labels.txt');
data = load(data_file);

%Add bias
data = [data , ones(size(data,1),1)]; 
labels = load(labels_file);
training_data_sizes = [200, 500, 800, 1000, 1500, 2000];

% Result vars
aucs = zeros(size(training_data_sizes));
accs = zeros(size(training_data_sizes));

%% Train the logistic regressor on different training data sizes & get accuracies
data_test = data(2001:4601,:);
labels_test = labels(2001:4601);
for i = 1:numel(training_data_sizes)
    disp(i);
    % Train the logistic regressor
    n = training_data_sizes(i);
    data_train = data(1:n,:);
    labels_train = labels(1:n);
    weights = logistic_train(data_train, labels_train);
    
    % Compute the predicted values and the testing accuracy
    predictions = sigmoid(data_test * weights);
    [~,~,~,aucs(i)] = perfcurve(labels_test, predictions,1);
    
    predictions = round(predictions);
    accs(i) = sum(predictions == labels_test)/numel(labels_test);

end


%% Plotting 
figure;
plot(training_data_sizes, accs, 'o-');
title('{\bf Logistic Regression}');
xlabel('n (Training data size)');
ylabel('Accuracy');

figure;
plot(training_data_sizes, aucs, 'o-');
title('{\bf Logistic Regression}');
xlabel('n (training data size)');
ylabel('AUC');
