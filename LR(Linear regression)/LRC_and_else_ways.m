clear;
% 1.人脸数据集的导入与数据处理框架
reshaped_faces=[];
% 声明数据库名
database_name = "ORL";

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
train_num_each = 5;% 每类训练数量
test_num_each = 5; % 每类测试数量
test_sum = test_num_each * people_num; % 测试总数
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
train_num_each = 4;% 每类训练数量
test_num_each = 6; % 每类测试数量
test_sum = test_num_each * people_num; % 测试总数
end

%FERET_80
if (database_name == "FERET")
    for i=1:80    
      for j=1:7       
        a=imread(strcat('C:\Users\hp\Desktop\face\FERET_80\ff',num2str(i),'_',num2str(j),'.tif'));              
        b = reshape(a,6400,1);
        b=double(b);        
        reshaped_faces=[reshaped_faces, b];  
      end
    end
row = 80;
column = 80;
people_num = 80;
pic_num_of_each = 7;
train_num_each = 4;% 每类训练数量
test_num_each = 3; % 每类测试数量
test_sum = test_num_each * people_num; % 测试总数
end

%Yale
if (database_name == "Yale")
    for i=1:15    
      for j=1:11
          if (i < 10)
            a=imread(strcat('C:\Users\hp\Desktop\face\face10080\subject0',num2str(i),'_',num2str(j),'.bmp')); 
          else
            a=imread(strcat('C:\Users\hp\Desktop\face\face10080\subject',num2str(i),'_',num2str(j),'.bmp'));  
          end
        b = reshape(a,8000,1);
        b=double(b);        
        reshaped_faces=[reshaped_faces, b];  
      end
    end
row = 100;
column = 80;
people_num = 15;
pic_num_of_each = 11;
train_num_each = 1;% 每类训练数量
test_num_each = 10; % 每类测试数量
test_sum = test_num_each * people_num; % 测试总数
end

%%PCA降维(新线性回归用到)

% % 求平均脸
% mean_face = mean(reshaped_faces,2);
% % 中心化
% centered_face = (reshaped_faces - mean_face);
% % 协方差矩阵
% cov_matrix = centered_face * centered_face';
% [eigen_vectors, dianogol_matrix] = eig(cov_matrix);
% % 从对角矩阵获取特征值
% eigen_values = diag(dianogol_matrix);
% % 对特征值按索引进行从大到小排序
% [sorted_eigen_values, index] = sort(eigen_values, 'descend'); 
% % 获取排序后的征值对应的特征向量
% sorted_eigen_vectors = eigen_vectors(:, index);
% 
%  % 取出相应数量特征脸(降到n维)
%    n = 200;
%    eigen_faces = sorted_eigen_vectors(:,1:n);
%     % 测试、训练数据降维
%     projected_data = eigen_faces' * (reshaped_faces - mean_face);
%     % 使用PCA降维
%      reshaped_faces = projected_data;
    
% 回归过程
dimension = row * column;
count_right = 0;

for i = 0:1:people_num - 1
    totest_index = i + 1; %取出图片对应标签
    %对每一类进行一次线性回归
    for k = train_num_each + 1:1:pic_num_of_each
       totest = reshaped_faces(:,i*pic_num_of_each + k); %取出每一待识别（分类）人脸
       distest = []; %记录距离
     for j = 0:1:people_num - 1
       batch_faces = reshaped_faces(:,j * pic_num_of_each + 1 :j * pic_num_of_each + pic_num_of_each); %取出每一类图片
       % 划分训练集与测试集
       %第一次  batch中的前train_num_each个数据作为训练集 后面数据作为测试集合
       train_data = batch_faces(:,1:train_num_each);
       test_data = batch_faces(:,train_num_each + 1:pic_num_of_each);
         % 1.线性回归
         w = inv(train_data' * train_data) * train_data' * totest;
         img_predict = train_data * w; % 计算预测图片           

         % 2.岭回归
%        rr_data = (train_data' * train_data) + eye(train_num_each)*10^-6;
%        w = inv(rr_data) * train_data' * totest;
%        img_predict = train_data * w; % 计算预测图片

         % 3.lasso回归
%        [B,FitInfo] = lasso(train_data , totest);
%        img_predict = train_data * B + FitInfo.Intercept;

         % 4.权重线性回归(代码有误)
%        W = eye(dimension);
%        kk = 10^-1;
%            for jj = 1:1:dimension
%               diff_data = reshaped_faces(j+1,:) - reshaped_faces(jj,:);
%               W(jj,jj) = exp((diff_data * diff_data')/(-2.0 * kk^2));
%            end
%            w = inv(train_data' * W * train_data) * train_data' * W * totest;

         % 5.新线性回归(已提前PCA降维)
%           rr_data = (train_data' * train_data) +
%           eye(train_num_each)*10^-6; 
%           w = inv(rr_data) * train_data' * test_data; % 改良w
%           img_predict = train_data * w; % 计算预测图片
         
       % show_face(img_predict,row,column); %预测人脸展示
         dis = img_predict - totest; % 计算误差
       
       distest = [distest,norm(dis)]; % 计算欧氏距离
     % 取出误差最小的预测图片 并找到他对应的标签 作为预测结果输出
     end
            [min_dis,label_index] = min(distest); % 找到最小欧氏距离下标（预测类）
            if label_index == totest_index
              count_right = count_right + 1;
            else  
                fprintf("预测错误：%d\n" ,(i + 1) * (k - train_num_each));
            end
    end
         
end
recognition_rate = count_right / test_sum; 

% 输入向量，显示脸
function fig = show_face(vector, row, column)
    fig = imshow(mat2gray(reshape(vector, [row, column])));
end
    

