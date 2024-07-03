%% 输出分类图像；
load('C:\Users\涂潮\Desktop\transform\PaviaU\PaviaU_gt.mat');
load('C:\Users\涂潮\Desktop\transform\PaviaU\reinfo_paviaU.mat');
%load('C:\Users\涂潮\Desktop\transform\PaviaU\result_2.mat')
% class = int8(data+1);
[row,col] = size(paviaU_gt);
class_size = 9;
re = (class_size+1)*ones(row,col);
for p = 1:size(class,1)
    re(reinfo(p,1),reinfo(p,2)) = class(p);
end
color = [255,0,0;0,128,128;255,165,0;255,127,20;255,255,0;255,20,147;0,255,0;0,100,0;0,255,255;0,0,0];%九类颜色
% color = [255,0,0;128,0,0;255,165,0;255,127,80;255,255,0;255,20,147;0,255,0;0,100,0;0,255,255;0,255,127;0,128,128;0,0,255;128,0,128;255,0,255;30,144,255;128,128,128;0,0,0];%十六类颜色
re = reshape(re,row*col,1);
result = color(re,:);
result = reshape(result,row,col,3);
imwrite(uint8(result),'C:\Users\涂潮\Desktop\transform\PaviaU\resultstan1.bmp');
figure;
imshow(uint8(result));

