clear all
close all
clc
insize = 1024;
image_num = 8;
sift1 = load(['D:\journal\re_do_from_ori\s4_kp_after_affine\SIFTflow\siftflowmap_', num2str(insize), '\', num2str(image_num),'sift1.mat']);
sift1 = sift1.sift1new;
sift2 = load(['D:\journal\re_do_from_ori\s4_kp_after_affine\SIFTflow\siftflowmap_', num2str(insize), '\', num2str(image_num),'sift2.mat']);
sift2 = sift2.sift2new;
sift1 = permute(sift1,[2,3,1]);
sift2 = permute(sift2,[2,3,1]);
SIFTcolor1=showColorSIFT(sift1);
SIFTcolor2=showColorSIFT(sift2);
catsift = zeros(insize*2, insize,3);
catsift(1:insize,1:insize,:)=SIFTcolor1;
catsift(insize+1:insize*2,1:insize,:)=SIFTcolor2;
a1 = SIFTcolor1(1,1,:);
a2 = SIFTcolor2(1,1,:);
figure,imshow(1-(SIFTcolor1-a1))
figure,imshow(1-(SIFTcolor2-a2))
imwrite(1-(SIFTcolor1-a1),'.\output\1.jpg');
imwrite(1-(SIFTcolor2-a2),'.\output\2.jpg');
%figure,imshow(catsift)