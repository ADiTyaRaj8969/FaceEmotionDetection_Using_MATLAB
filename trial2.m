load('model.mat'); 

% Create a video object to capture video from the webcam
cam = webcam; % Use the default webcam

% Create a face detector object
faceDetector = vision.CascadeObjectDetector();

% Define the emotion labels (adjust according to your model)
emotionLabels = {'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'};

% Create a figure for displaying the video
figure;

% Start capturing video
while true
    % Capture a frame
    img = snapshot(cam);
    
    % Detect faces in the image
    bbox = step(faceDetector, img);
    
    % If faces are detected
    if ~isempty(bbox)
        % Loop through each detected face
        for i = 1:size(bbox, 1)
            % Extract the face region
            face = imcrop(img, bbox(i, :));
            face = imresize(face, [48 48]); % Resize to match the input size of the model
            face = rgb2gray(face); % Convert to grayscale
            face = double(face) / 255; % Normalize the image
            
            % Reshape the face for classification (add batch dimension)
            face = reshape(face, [48, 48, 1, 1]); % Reshape to [height, width, channels, batch_size]
            
            % Classify the emotion
            emotion = classify(net, face);
            
            % Convert the classified emotion to a string
            label = char(emotion);
            
            % Check if the classified emotion is in the emotionLabels
            if ismember(label, emotionLabels)
                % Display the detected emotion on the image
                img = insertObjectAnnotation(img, 'rectangle', bbox(i, :), label);
            else
                % Default label if no match is found
                img = insertObjectAnnotation(img, 'rectangle', bbox(i, :), 'Unknown');
            end
        end
    end
    
    % Display the annotated image
    imshow(img);
    title('Face Emotion Detection');
    
    % Break the loop on key press (specifically 'q' to quit)
    if ~isempty(get(gcf, 'CurrentCharacter')) && strcmp(get(gcf, 'CurrentCharacter'), 'q')
        break;
    end
end

% Clean up
clear cam;