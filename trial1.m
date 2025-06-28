% Clear the workspace
clc;
clear;

% Load the dataset
% Assumes the dataset is organized as 'dataset/emotion_name/image_files.jpg'
datasetPath = 'archive'; % Replace with your dataset path
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Display dataset info
disp('Dataset loaded successfully:');
disp(countEachLabel(imds));

% Split dataset into training and validation sets
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');

% Define input image size and number of emotion categories
inputSize = [48 48 1]; % Assuming grayscale 48x48 images
numClasses = numel(unique(imds.Labels));

% Define the CNN architecture
layers = [
    imageInputLayer(inputSize, 'Name', 'input', 'Normalization', 'none')
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'batchnorm1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1')

    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'batchnorm2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2')

    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'batchnorm3')
    reluLayer('Name', 'relu3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool3')

    fullyConnectedLayer(256, 'Name', 'fc1')
    batchNormalizationLayer('Name', 'batchnorm4')
    reluLayer('Name', 'relu4')
    dropoutLayer(0.7, 'Name', 'dropout1')
    fullyConnectedLayer(numClasses, 'Name', 'fc2')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

% Set training options
options = trainingOptions('adam', ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 64, ...
    'InitialLearnRate', 0.0005, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', imdsValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Train the network
disp('Training started...');
net = trainNetwork(imdsTrain, layers, options);
disp('Training complete!');

% Save the trained model along with labels
emotionLabels = categories(imdsTrain.Labels); % Extract unique emotion labels
modelPath = 'emotionNet.mat';
save(modelPath, 'net', 'emotionLabels');
disp(['Model and labels saved to ', modelPath]);

% Test the model on a few validation images
disp('Evaluating on validation data...');
YPred = classify(net, imdsValidation);
YValidation = imdsValidation.Labels;

% Calculate accuracy
accuracy = sum(YPred == YValidation) / numel(YValidation);
disp(['Validation Accuracy: ', num2str(accuracy * 100), '%']);