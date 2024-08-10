%% clean up

close gcf
% closes all figures
close all
% clears the workspace
clear all
% clears the command window
clc

rng(42);
train_data = readmatrix("D:\Work\City University of London\Neural Computing\Project\data\predictive_maintenance_training.csv");
[X, Y] = prepare_data(train_data);
disp("Loaded training data...")

test_data = readmatrix("D:\Work\City University of London\Neural Computing\Project\data\predictive_maintenance_test.csv");
[X_test, Y_test] = prepare_data(test_data);
Y_test = Y_test(:, 2) > Y_test(:, 1);
disp("Loaded test data...")


print_output = 0;

neuron_tries = [2 3 4 5 6];
neuron_list_length = size(neuron_tries, 2);

epochs_tries = [10 25 50 100 200];
epoch_list_length = size(epochs_tries, 2);

%lr_tries = [0.01 0.011 0.012 0.013 0.014 0.015 0.016 0.017 0.018 0.019 0.02];
lr_tries = [0.05 0.06 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0];
lr_list_length = size(lr_tries, 2);

accuracy_list = zeros(neuron_list_length, epoch_list_length, lr_list_length);
accuracy_list = zeros(epoch_list_length, lr_list_length);
accuracy_list = zeros(1, epoch_list_length);
time_list = zeros(epoch_list_length, lr_list_length);
best_accuracy = -1;
best_training_time = 10e10;
best_lr = 0;
best_n_epochs = 0;

%for i = 1:neuron_list_length
for j = 1:epoch_list_length
    %for k = 1:lr_list_length
        current_n_epochs = epochs_tries(j);
        current_lr = 0.3;
        [trained_weights, ~, training_time] = train_network(X, Y, 2, current_n_epochs, current_lr, print_output);
        accuracy = evaluate_accuracy(X_test, Y_test, trained_weights);
        accuracy = accuracy * 100;
        accuracy_list(j) = accuracy;

        if (accuracy > best_accuracy)
            best_accuracy = accuracy;
            best_training_time = training_time;
            best_lr = current_lr;
            best_n_epochs = current_n_epochs;
            best_weights = trained_weights;
        end
    %end
end
%end

disp(accuracy_list)
%disp(best_accuracy)
%disp(best_training_time)
%disp(best_lr)
%disp(best_n_epochs)

%%

experiment_size = 100;
training_time_experiment = zeros(1, experiment_size);
for i = 1:experiment_size
    [~, ~, training_time] = train_network(X, Y, 2, best_n_epochs, best_lr, print_output);
    training_time_experiment(i) = training_time;
end
avg_training_time = mean(training_time_experiment);
disp(["Avg training time:" avg_training_time])
%accuracy = evaluate_accuracy(X_test, Y_test, trained_weights);

%%

print_output = 0;

epochs_tries = [10 25 50 100 200];
epoch_list_length = size(epochs_tries, 2);

accuracy_list = zeros(1, epoch_list_length);
time_list = zeros(epoch_list_length, lr_list_length);
best_accuracy = -1;
best_training_time = 10e10;
best_lr = 0;
best_n_epochs = 0;

for j = 1:epoch_list_length
        current_n_epochs = epochs_tries(j);
        current_lr = 0.3;
        [trained_weights, ~, training_time] = train_network(X, Y, 2, current_n_epochs, current_lr, print_output);
        accuracy = evaluate_accuracy(X_test, Y_test, trained_weights);
        accuracy = accuracy * 100;
        accuracy_list(j) = accuracy;
end

disp(accuracy_list)
%%

function [X, Y] = prepare_data(dataset)
X = dataset;
Y = X(:, 7)';
Y = Y + 1;
Y = ind2vec(Y);
X(:, 7) = [];
Y = Y';
end

function [trained_weights, loss_history, training_time] = train_network(X, Y, num_hidden, num_epochs, learning_rate, print_output)
% Implements a feed-forward backpropagation neural network for binary classification in MATLAB.
%
% Arguments:
% - X: input data, a matrix of size (num_samples, num_inputs)
% - Y: target labels, a matrix of size (num_samples, num_classes)
% - num_hidden: number of neurons in the hidden layer
% - num_epochs: number of epochs for training
% - learning_rate: learning rate for stochastic gradient descent
%
% Returns:
% - trained_weights: a cell array of trained weight matrices for the network
% - loss_history: a vector of losses at each epoch during training
% - training_time: the time taken to train the network


relu = @(x) max(0, x);
relu_derivative = @(x) (x > 0);
sigmoid = @(x) 1./(1+exp(-x));
% Initialize weights
num_inputs = size(X, 2);
num_classes = size(Y, 2);
W1 = randn(num_inputs, num_hidden);
b1 = zeros(1, num_hidden);
W2 = randn(num_hidden, num_classes);
b2 = zeros(1, num_classes);
if print_output
    disp("Initialized network...")
end

% Training loop
loss_history = zeros(num_epochs, 1);
tic;
for epoch = 1:num_epochs
    % Forward pass
    Z1 = X * W1 + b1;
    A1 = relu(Z1); % ReLU activation function
    Z2 = A1 * W2 + b2;
    Y_pred = sigmoid(Z2); % Add epsilon to ensure strictly positive output


    % Compute loss
    loss = -sum(sum(Y .* log(Y_pred))) / size(X, 1);
    loss_history(epoch) = loss;

    % Backward pass
    dZ2 = Y_pred - Y;
    dW2 = A1' * dZ2 / size(X, 1);
    db2 = sum(dZ2) / size(X, 1);
    dA1 = dZ2 * W2';
    dZ1 = dA1 .* relu_derivative(Z1); % ReLU derivative
    dW1 = X' * dZ1 / size(X, 1);
    db1 = sum(dZ1) / size(X, 1);

    % Update weights
    W2 = W2 - learning_rate * dW2;
    b2 = b2 - learning_rate * db2;
    W1 = W1 - learning_rate * dW1;
    b1 = b1 - learning_rate * db1;

    % Display progress and loss every 10 epochs
    if mod(epoch, 100) == 0 && print_output
        fprintf('Epoch %d/%d\n', epoch, num_epochs);
    end
end
training_time = toc;
% Return trained weights and loss history
trained_weights = {W1, b1, W2, b2};
end


function Y_pred = predict(X, trained_weights)
% Predicts the output labels of a given input data using a trained neural network.
%
% Arguments:
% - X: input data, a matrix of size (num_samples, num_inputs)
% - trained_weights: a cell array of trained weight matrices for the network
%
% Returns:
% - Y_pred: predicted output labels, a matrix of size (num_samples, num_classes)

relu = @(x) max(0, x);
sigmoid = @(x) 1./(1+exp(-x));
% Unpack trained weights
W1 = trained_weights{1};
b1 = trained_weights{2};
W2 = trained_weights{3};
b2 = trained_weights{4};

% Forward pass
Z1 = X * W1 + b1;
A1 = relu(Z1); % ReLU activation function
Z2 = A1 * W2 + b2;
Y_pred = sigmoid(Z2); % ReLU activation function

% Convert predicted labels to binary classes
Y_pred = Y_pred(:, 2) > Y_pred(:, 1);
end

function accuracy = evaluate_accuracy(X_test, Y_test, trained_weights)
% Evaluates the accuracy of a given neural network on a test set.
%
% Arguments:
% - X_test: test input data, a matrix of size (num_samples, num_inputs)
% - Y_test: test target labels, a matrix of size (num_samples, num_classes)
% - trained_weights: a cell array of trained weight matrices for the network
%
% Returns:
% - accuracy: the accuracy of the network on the test set

% Predict output labels for test set
Y_pred = predict(X_test, trained_weights);

% Calculate accuracy
accuracy = sum(Y_pred == Y_test) / length(Y_test);
end

