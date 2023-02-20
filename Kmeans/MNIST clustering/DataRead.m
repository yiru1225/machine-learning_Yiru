function DataRead()





load ./train_images.mat    % Read image data

load ./train_labels.mat       %Read lable of images

ImgNum = 1; 
GetOneImg = train_images(:,:,ImgNum);
figure(1);
imshow(GetOneImg,[ ]);   %Show the image


 disp(['The number of Image is :  ',num2str(train_labels(ImgNum))]);
end












