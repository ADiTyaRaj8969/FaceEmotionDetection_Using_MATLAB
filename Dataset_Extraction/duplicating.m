clear all;
close all;
clc;

image_array = dir('Emotions_Dataset/Neutral/*.jpg');
L=length(image_array);
for i = 1:L
    disp(i)
    disp(image_array(i).name);
    str = strcat('Emotions_Dataset/Neutral/',image_array(i).name);
    I =imread(str);
    if(size(I,3)==3)
        I = rgb2gray(I);
    end
    I_res = cat(3,I,I,I);
    imwrite(I_res,str)
end

image_array = dir('Emotions_Dataset/Happy/*.jpg');
L=length(image_array);
for i = 1:L
    disp(i)
    disp(image_array(i).name);
    str = strcat('Emotions_Dataset/Happy/',image_array(i).name);
    I =imread(str);
    if(size(I,3)==3)
        I = rgb2gray(I);
    end
    I_res = cat(3,I,I,I);
    imwrite(I_res,str)
end

image_array = dir('Emotions_Dataset/Surprise/*.jpg');
L=length(image_array);
for i = 1:L
    disp(i)
    disp(image_array(i).name);
    str = strcat('Emotions_Dataset/Surprise/',image_array(i).name);
    I =imread(str);
    if(size(I,3)==3)
        I = rgb2gray(I);
    end
    I_res = cat(3,I,I,I);
    imwrite(I_res,str)
end

image_array = dir('Emotions_Dataset/Fear/*.jpg');
L=length(image_array);
for i = 1:L
    disp(i)
    disp(image_array(i).name);
    str = strcat('Emotions_Dataset/Fear/',image_array(i).name);
    I =imread(str);
    if(size(I,3)==3)
        I = rgb2gray(I);
    end
    I_res = cat(3,I,I,I);
    imwrite(I_res,str)
end

image_array = dir('Emotions_Dataset/Angry/*.jpg');
L=length(image_array);
for i = 1:L
    disp(i)
    disp(image_array(i).name);
    str = strcat('Emotions_Dataset/Angry/',image_array(i).name);
    I =imread(str);
    if(size(I,3)==3)
        I = rgb2gray(I);
    end
    I_res = cat(3,I,I,I);
    imwrite(I_res,str)
end

image_array = dir('Emotions_Dataset/Sad/*.jpg');
L=length(image_array);
for i = 1:L
    disp(i)
    disp(image_array(i).name);
    str = strcat('Emotions_Dataset/Sad/',image_array(i).name);
    I =imread(str);
    if(size(I,3)==3)
        I = rgb2gray(I);
    end
    I_res = cat(3,I,I,I);
    imwrite(I_res,str)
end

image_array = dir('Emotions_Dataset/Disgust/*.jpg');
L=length(image_array);
for i = 1:L
    disp(i)
    disp(image_array(i).name);
    str = strcat('Emotions_Dataset/Disgust/',image_array(i).name);
    I =imread(str);
    if(size(I,3)==3)
        I = rgb2gray(I);
    end
    I_res = cat(3,I,I,I);
    imwrite(I_res,str)
end
