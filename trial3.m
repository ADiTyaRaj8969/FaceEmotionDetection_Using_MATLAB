% Load the custom-trained emotion detection model
load('emotionDetectionCustomModel.mat', 'trainedNet');

% Define the emotion labels (adjust according to your dataset)
emotionLabels = {'anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'contempt'};

% Create a webcam object (use webcam(1) for specific cameras)
cam = webcam;

% Load the face detector (Haar Cascade)
faceDetector = vision.CascadeObjectDetector();

% Start live webcam feed for emotion detection
figure;

while true
    % Capture a frame from the webcam
    img = snapshot(cam);

    % Detect faces in the captured image
    bbox = step(faceDetector, img);
    
    % If no face is detected, continue to the next frame
    if isempty(bbox)
        subplot(1, 2, 1);
        imshow(img);
        title('Live Feed');
        
        subplot(1, 2, 2);
        imshow(zeros(size(img, 1), size(img, 2)), []);
        title('Grayscale Face (No Face Detected)');
        
        drawnow;
        continue;
    end
    
    % For each detected face, extract the face region
    for i = 1:size(bbox, 1)
        face = imcrop(img, bbox(i, :));  % Crop the face
        
        % Resize the cropped face to match the input size of the custom CNN (64x64)
        faceResized = imresize(face, [64 64]);
        
        % Convert the cropped face to grayscale if it is RGB
        if size(faceResized, 3) == 3
            faceResizedGray = rgb2gray(faceResized);  % Convert to grayscale
        else
            faceResizedGray = faceResized;
        end
        
        % Classify the emotion using the trained custom CNN model
        [label, scores] = classify(trainedNet, faceResizedGray);
        
        % Display the predicted emotion on the image
        predictedEmotion = emotionLabels{double(label)};
        img = insertText(img, bbox(i, 1:2), predictedEmotion, 'FontSize', 18, 'BoxColor', 'yellow', 'BoxOpacity', 0.6, 'TextColor', 'black');
        
        % Show the grayscale face image
        subplot(1, 2, 2);
        imshow(faceResizedGray, []);
        title(['Grayscale Face (', predictedEmotion, ')']);
    end
    
    % Display the live feed with emotion prediction
    subplot(1, 2, 1);
    imshow(img);
    title('Live Feed');
    
    % Pause briefly to prevent high CPU usage
    drawnow;
end

% Clean up
clear cam;
