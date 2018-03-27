clear
load('ad_data.mat');

% Add bias
X_train = [X_train ones(size(X_train, 1), 1) ];
X_test = [X_test ones(size(X_test, 1), 1) ];

pars = [1e-8, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];

% Result vars
feature_num = zeros(size(pars));
aucs = zeros(size(pars));

% Perform the experiment.
for i=1:numel(pars)
    % Train the logistic regressor
    [weights, bias] = logistic_l1_train(X_train, y_train, pars(i));
    
    % Compute the predicted values and performance
    feature_num(i) = sum( weights~= 0);
    predictions = X_test * weights;
    [~, ~, threshold, aucs(i)] = perfcurve(y_test, predictions, 1);
end

%% Plotting
figure
plot(pars, feature_num, '-o')
title('{\bf Count of Non-zero Weights vs. Regularization Parameter}')
xlabel('Regularization Parameter')
ylabel('Number of non-zero weight')

figure
plot(pars, aucs, '-o')
title('{\bf Area Under Curve vs. Regularization Parameter}')
xlabel('Regularization Parameter')
ylabel('AUC')
