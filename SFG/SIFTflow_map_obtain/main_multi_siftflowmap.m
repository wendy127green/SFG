clear all
close all
clc
pairflag=importdata('D:\journal\for_submit\matrix_sequence_manual_validation.csv');
for image_num=0:480
    if ~isempty(strfind(pairflag{image_num+2},'training'))
        for mapsize = [32 128]
            im10=imread(['D:\journal\re_do_from_ori\s3_affine_result_obtain\zoom\512after_affine\', num2str(image_num),'_1.jpg']);
            im20=imread(['D:\journal\re_do_from_ori\s3_affine_result_obtain\zoom\512after_affine\', num2str(image_num),'_2.jpg']);
           
            if mapsize == 128
                im1 = imfilter(im10,fspecial('gaussian',7,0.5),'same','replicate');
                im2 = imfilter(im20,fspecial('gaussian',7,0.5),'same','replicate');
            end
            if mapsize == 32
                im1 = imfilter(im10,fspecial('gaussian',11,0.5),'same','replicate');
                im2 = imfilter(im20,fspecial('gaussian',11,0.5),'same','replicate');
            end
            
            im1=imresize(im1, mapsize/512, 'bicubic');
            im2=imresize(im2, mapsize/512, 'bicubic');

            im1=im2double(im1);
            im2=im2double(im2);

            %figure;imshow(im1);figure;imshow(im2);

            cellsize=3;
            gridspacing=1;

            addpath(fullfile(pwd,'mexDenseSIFT'));
            addpath(fullfile(pwd,'mexDiscreteFlow'));

            sift1new = mexDenseSIFT(im1,cellsize,gridspacing);
            sift2new = mexDenseSIFT(im2,cellsize,gridspacing);
            sift1new = permute(sift1new,[3,1,2]);
            sift2new = permute(sift2new,[3,1,2]);
            save(['.\siftflowmap_',num2str(mapsize),'\',num2str(image_num),'sift1.mat'],'sift1new') 
            save(['.\siftflowmap_',num2str(mapsize),'\',num2str(image_num),'sift2.mat'],'sift2new') 

        end
    end
end
