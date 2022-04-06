clear all
close all
clc
for image_num=1:480
    im1=imread(['D:\journal\for_submit\s3_affine_result_obtain\zoom\512_all_480samples\',num2str(image_num),'_1.jpg']);
    im2=imread(['D:\journal\for_submit\s3_affine_result_obtain\zoom\512_all_480samples\',num2str(image_num),'_2.jpg']);

    %im1=imresize(imfilter(im1,fspecial('gaussian',7,1.),'same','replicate'),0.5,'bicubic');
    %im2=imresize(imfilter(im2,fspecial('gaussian',7,1.),'same','replicate'),0.5,'bicubic');
    im1=imfilter(im1,fspecial('gaussian',7,1.),'same','replicate');
    im2=imfilter(im2,fspecial('gaussian',7,1.),'same','replicate');

    im1=im2double(im1);
    im2=im2double(im2);

    %figure;imshow(im1);figure;imshow(im2);

    cellsize=3;
    gridspacing=1;

    addpath(fullfile(pwd,'mexDenseSIFT'));
    addpath(fullfile(pwd,'mexDiscreteFlow'));

    sift1new = mexDenseSIFT(im1,cellsize,gridspacing);
    sift2new = mexDenseSIFT(im2,cellsize,gridspacing);

    save(['D:\journal\for_submit\s5_kp\SIFTflow\outsift_512\',num2str(image_num),'sift1.mat'],'sift1new') 
    save(['D:\journal\for_submit\s5_kp\SIFTflow\outsift_512\',num2str(image_num),'sift2.mat'],'sift2new') 
end
%{
%% sift 3D visualization by gelin
s1 = sift1;
size1 = size(s1);
s1 = reshape(s1,size1(1)*size1(2),128);
s1 = double(s1);
[pc,score,latent,tsquare] = pca(s1);
%cumsum(latent)./sum(latent)
s1= bsxfun(@minus,s1,mean(s1,1));
r1 = s1*pc(:,1:3);
r1 = reshape(r1,size1(1),size1(2),3);
s2 = sift2;
size2 = size(s2);
s2 = reshape(s2,size2(1)*size2(2),128);
s2 = double(s2);
[pc,score,latent,tsquare] = pca(s2);
%cumsum(latent)./sum(latent)
s2= bsxfun(@minus,s2,mean(s2,1));
r2 = s2*pc(:,1:3);
r2 = reshape(r2,size2(1),size2(2),3);
r1 = r1 - min(r1(:));
r1 = r1 / max(r1(:));
r2 = r2 - min(r2(:));
r2 = r2 / max(r2(:));
r1 = r1*255;
r2 = r2*255;
figure,subplot(221)
imshow(im1);
subplot(222);
imshow(im2);
subplot(223);
imshow(uint8(r1));
subplot(224);
imshow(uint8(r2));

%}

%{
SIFTflowpara.alpha=2*255;
SIFTflowpara.d=40*255;
SIFTflowpara.gamma=0.005*255;
SIFTflowpara.nlevels=4;
SIFTflowpara.wsize=2;
SIFTflowpara.topwsize=10;
SIFTflowpara.nTopIterations = 60;
SIFTflowpara.nIterations= 30;

tic;[vx,vy,energylist]=SIFTflowc2f(sift1,sift2,SIFTflowpara);toc

warpI2=warpImage(im2,vx,vy);
figure;
subplot(221)
imshow(im1);
subplot(222)
imshow(im2);
subplot(223)
imshow(im1);
subplot(224)
imshow(warpI2);

% display flow
clear flow;
flow(:,:,1)=vx;
flow(:,:,2)=vy;
figure;imshow(flowToColor(flow));

return;

% this is the code doing the brute force matching
tic;[flow2,energylist2]=mexDiscreteFlow(Sift1,Sift2,[alpha,alpha*20,60,30]);toc
figure;imshow(flowToColor(flow2));
%}
