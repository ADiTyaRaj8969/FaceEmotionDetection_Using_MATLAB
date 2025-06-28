clear all;
close all;
clc;

image_array = dir('Emotions_Dataset/Neutral/*.png');
L=length(image_array)


for i = 1:L
    disp(i)
    disp(image_array(i).name);
    str = strcat('Emotions_Dataset/Neutral/',image_array(i).name);
    I =imread(str);
    I_res = imresize(I,[227,227]);
    imwrite(I_res,str)
end

image_array = dir('Emotions_Dataset/Sad/*.png');
L=length(image_array)


for i = 1:L
    disp(i)
    disp(image_array(i).name);
    str = strcat('Emotions_Dataset/Sad/',image_array(i).name);
    I =imread(str);
    I_res = imresize(I,[227,227]);
    imwrite(I_res,str)
end

image_array = dir('Emotions_Dataset/Surprise/*.png');
L=length(image_array)


for i = 1:L
    disp(i)
    disp(image_array(i).name);
    str = strcat('Emotions_Dataset/Surprise/',image_array(i).name);
    I =imread(str);
    I_res = imresize(I,[227,227]);
    imwrite(I_res,str)
end
 
image_array = dir('Emotions_Dataset/Angry/*.png');
L=length(image_array)


for i = 1:L
    disp(i)
    disp(image_array(i).name);
    str = strcat('Emotions_Dataset/Angry/',image_array(i).name);
    I =imread(str);
    I_res = imresize(I,[227,227]);
    imwrite(I_res,str)
end

image_array = dir('Emotions_Dataset/Disgust/*.png');
L=length(image_array)


for i = 1:L
    disp(i)
    disp(image_array(i).name);
    str = strcat('Emotions_Dataset/Disgust/',image_array(i).name);
    I =imread(str);
    I_res = imresize(I,[227,227]);
    imwrite(I_res,str)
end

image_array = dir('Emotions_Dataset/Happy/*.png');
L=length(image_array)


for i = 1:L
    disp(i)
    disp(image_array(i).name);
    str = strcat('Emotions_Dataset/Happy/',image_array(i).name);
    I =imread(str);
    I_res = imresize(I,[227,227]);
    imwrite(I_res,str)
end

image_array = dir('Emotions_Dataset/Fear/*.png');
L=length(image_array)


for i = 1:L
    disp(i)
    disp(image_array(i).name);
    str = strcat('Emotions_Dataset/Fear/',image_array(i).name);
    I =imread(str);
    I_res = imresize(I,[227,227]);
    imwrite(I_res,str)
end
