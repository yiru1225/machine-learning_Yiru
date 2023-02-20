clear;
% 1.人脸数据集的导入与数据处理框架
reshaped_faces=[];
% 声明数据库名
database_name = "AR";

% ORL5646
if (database_name == "ORL")
  for i=1:40    
    for j=1:10       
        if(i<10)
           a=imread(strcat('C:\Users\hp\Desktop\face\ORL56_46\orl',num2str(i),'_',num2str(j),'.bmp'));     
        else
            a=imread(strcat('C:\Users\hp\Desktop\face\ORL56_46\orl',num2str(i),'_',num2str(j),'.bmp'));  
        end          
        b = reshape(a,2576,1);
        b=double(b);        
        reshaped_faces=[reshaped_faces, b];  
    end
  end
row = 56; 
column = 46;
people_num = 40;
pic_num_of_each = 10;
end

%AR5040
if (database_name == "AR")
    for i=1:40    
      for j=1:10       
        if(i<10)
           a=imread(strcat('C:\AR_Gray_50by40\AR00',num2str(i),'-',num2str(j),'.tif'));     
        else
            a=imread(strcat('C:\AR_Gray_50by40\AR0',num2str(i),'-',num2str(j),'.tif'));  
        end          
        b = reshape(a,2000,1);
        b=double(b);        
        reshaped_faces=[reshaped_faces, b];  
      end
    end
row = 50;
column = 40;
people_num = 40;
pic_num_of_each = 10;

end

%COIL20（玩具数据集）
if (database_name == "COIL")
    for i=1:20    
      for j=0:71       
        a=imread(strcat('C:\Users\hp\Desktop\face\coil-20-proc\obj',num2str(i),'__',num2str(j),'.png'));              
        b = reshape(a,16384,1);
        b=double(b);        
        reshaped_faces=[reshaped_faces, b];  
      end
    end
row = 128;
column = 128;
people_num = 20;
pic_num_of_each = 72;
end

dimension = row * column;
K = 2; % 设置K-means的K
% K-means训练
test_data = reshaped_faces(:,1:pic_num_of_each * K);
[idx,center] = kmeans(test_data',K); %idx是分类类别，center是质心集

%LDA过程
% 算每个类的平均
k = 1; 
class_mean = zeros(dimension, people_num); 
for i=1:people_num
    % 求一列（即一个人）的均值
    temp = class_mean(:,i);
    % 遍历每个人的train_pic_num_of_each张用于训练的脸，相加算平均
    for j=1:pic_num_of_each
        temp = temp + reshaped_faces(:,k);
        k = k + 1;
    end
    class_mean(:,i) = temp / pic_num_of_each;
end

% 算类类间散度矩阵Sb
Sb = zeros(dimension, dimension);
all_mean = mean(reshaped_faces, 2); % 全部的平均
for i=1:people_num
    % 以每个人的平均脸进行计算，这里减去所有平均，中心化
    centered_data = class_mean(:,i) - all_mean;
    Sb = Sb + centered_data * centered_data';
end
Sb = Sb / people_num;

% 算类内散度矩阵Sw
Sw = zeros(dimension, dimension);
k = 1; % p表示每一张图片
for i=1:people_num % 遍历每一个人
    for j=1:pic_num_of_each % 遍历一个人的所有脸计算后相加
        centered_data = reshaped_faces(:,k) - class_mean(:,i);
        Sw = Sw + centered_data * centered_data';
        k = k + 1;
    end
end
Sw = Sw / (people_num * pic_num_of_each);

% 目标函数一：经典LDA（伪逆矩阵代替逆矩阵防止奇异值）
 target = pinv(Sw) * Sb;
 % 求特征值、特征向量
[eigen_vectors, dianogol_matrix] = eig(target);
eigen_values = diag(dianogol_matrix);

% 对特征值、特征向量进行排序
[sorted_eigen_values, index] = sort(eigen_values, 'descend'); 
eigen_vectors = eigen_vectors(:, index);

% 降维与可视化
class_num_to_show = K;
pic_num_in_a_class = pic_num_of_each;
pic_to_show = class_num_to_show * pic_num_in_a_class;
m = 2; % 制定可视化维数
% 取出相应数量特征向量
    project_matrix = eigen_vectors(:,1:m);
    % 投影
    projected_test_data = project_matrix' * (reshaped_faces - all_mean);
    projected_test_data = projected_test_data(:,1:pic_to_show);
    pattern = projected_test_data';

%可视化
if(m ==2)
figure;
[max_xy,index]=max(pattern); %用于在图像上标记未聚类原类别
for i = 1 : pic_num_of_each * K
    if(i <= pic_num_of_each)
        if idx(i,1) == 1 
         scatter(pattern(i,1),pattern(i,2),'o','r*');
    elseif idx(i,1) == 2
         scatter(pattern(i,1),pattern(i,2),'o','g*');
    elseif idx(i,1) == 3
         scatter(pattern(i,1),pattern(i,2),'o','b*');
    elseif idx(i,1) == 4
         scatter(pattern(i,1),pattern(i,2),'o','y*');
        end
    elseif(i <= pic_num_of_each * 2)
        if idx(i,1) == 1 
         scatter(pattern(i,1),pattern(i,2),'^','r*');
    elseif idx(i,1) == 2
         scatter(pattern(i,1),pattern(i,2),'^','g*');
    elseif idx(i,1) == 3
         scatter(pattern(i,1),pattern(i,2),'^','b*');
    elseif idx(i,1) == 4
         scatter(pattern(i,1),pattern(i,2),'^','y*');
        end
    elseif(i <= pic_num_of_each * 3)
        if idx(i,1) == 1 
         scatter(pattern(i,1),pattern(i,2),'x','r*');
    elseif idx(i,1) == 2
         scatter(pattern(i,1),pattern(i,2),'x','g*');
    elseif idx(i,1) == 3
         scatter(pattern(i,1),pattern(i,2),'x','b*');
    elseif idx(i,1) == 4
         scatter(pattern(i,1),pattern(i,2),'x','y*');
        end
    end 
hold on;
end
text(max_xy(1,1)-10,max_xy(1,2),'第一类：o');
text(max_xy(1,1)-10,max_xy(1,2)-15,'第二类：▲');
if(K==3) 
    text(max_xy(1,1)-10,max_xy(1,2)-30,'第三类：x');
end
end

if(m==3)
figure
[max_xyz,index]=max(pattern); %用于在图像上标记未聚类原类别
for i = 1 :pic_num_of_each * K
    if(i <= pic_num_of_each)
         if idx(i,1) == 1 
         scatter3(pattern(i,1),pattern(i,2),pattern(i,3),'o','r*');
    elseif idx(i,1) == 2
         scatter3(pattern(i,1),pattern(i,2),pattern(i,3),'o','g*');
    elseif idx(i,1) == 3
         scatter3(pattern(i,1),pattern(i,2),pattern(i,3),'o','b*');
    elseif idx(i,1) == 4
         scatter3(pattern(i,1),pattern(i,2),pattern(i,3),'o','y*');
         end
    elseif(i <= pic_num_of_each * 2)
         if idx(i,1) == 1 
         scatter3(pattern(i,1),pattern(i,2),pattern(i,3),'^','r*');
    elseif idx(i,1) == 2
         scatter3(pattern(i,1),pattern(i,2),pattern(i,3),'^','g*');
    elseif idx(i,1) == 3
         scatter3(pattern(i,1),pattern(i,2),pattern(i,3),'^','b*');
    elseif idx(i,1) == 4
         scatter3(pattern(i,1),pattern(i,2),pattern(i,3),'^','y*');
         end
    elseif(i <= pic_num_of_each * 3)
         if idx(i,1) == 1 
         scatter3(pattern(i,1),pattern(i,2),pattern(i,3),'x','r*');
    elseif idx(i,1) == 2
         scatter3(pattern(i,1),pattern(i,2),pattern(i,3),'x','g*');
    elseif idx(i,1) == 3
         scatter3(pattern(i,1),pattern(i,2),pattern(i,3),'x','b*');
    elseif idx(i,1) == 4
         scatter3(pattern(i,1),pattern(i,2),pattern(i,3),'x','y*');
         end
    end 
    hold on;
end    
text(max_xyz(1,1)-10,max_xyz(1,2),max_xyz(1,3),'第一类：o');
text(max_xyz(1,1)-10,max_xyz(1,2)-15,max_xyz(1,3)-15,'第二类：▲');
if(K==3)
    text(max_xyz(1,1)-10,max_xyz(1,2)-30,max_xyz(1,3)-30,'第三类：x');
end
end

% 输入向量，显示脸
function fig = show_face(vector, row, column)
    fig = mat2gray(reshape(vector, [row, column]));
end
