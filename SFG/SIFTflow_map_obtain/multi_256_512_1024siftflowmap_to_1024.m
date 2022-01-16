clear all
close all
clc
pairflag=importdata('D:\journal\for_submit\matrix_sequence_manual_validation.csv');
outsize = 1024;
dim = 128;
for image_num=0:480
    if ~isempty(strfind(pairflag{image_num+2},'training'))
       %% 256 to 1024 
        insize = 256;
        sift1 = load(['.\siftflowmap_', num2str(insize), '\', num2str(image_num),'sift1.mat']);
        sift1 = sift1.sift1new;
        sift2 = load(['.\siftflowmap_', num2str(insize), '\', num2str(image_num),'sift2.mat']);
        sift2 = sift2.sift2new;
        sift1_256to1024 = gl_multi_dimention_bilinear_interpolation(sift1, insize, outsize, dim);
        sift2_256to1024 = gl_multi_dimention_bilinear_interpolation(sift2, insize, outsize, dim);
        %% 512 to 1024
        insize = 512;
        sift1 = load(['.\siftflowmap_', num2str(insize), '\', num2str(image_num),'sift1.mat']);
        sift1 = sift1.sift1new;
        sift2 = load(['.\siftflowmap_', num2str(insize), '\', num2str(image_num),'sift2.mat']);
        sift2 = sift2.sift2new;
        sift1_512to1024 = gl_multi_dimention_bilinear_interpolation(sift1, insize, outsize, dim);
        sift2_512to1024 = gl_multi_dimention_bilinear_interpolation(sift2, insize, outsize, dim);
        %% 1024 load
        insize = 1024;
        sift1 = load(['.\siftflowmap_', num2str(insize), '\', num2str(image_num),'sift1.mat']);
        sift1_1024 = sift1.sift1new;
        sift2 = load(['.\siftflowmap_', num2str(insize), '\', num2str(image_num),'sift2.mat']);
        sift2_1024 = sift2.sift2new;
        %% combine and save
        sift1new = cat(1,sift1_256to1024, sift1_512to1024, sift1_1024);
        sift2new = cat(1,sift2_256to1024, sift2_512to1024, sift2_1024);
        
        save(['.\multi_siftflowmap_',num2str(outsize),'\',num2str(image_num),'sift1.mat'],'sift1new') 
        save(['.\multi_siftflowmap_',num2str(outsize),'\',num2str(image_num),'sift2.mat'],'sift2new') 
        
      
    end
end