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
train_pic_num_of_each = 7; % 每张人脸训练数量
test_pic_num_of_each = 3;  % 每张人脸测试数量
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
train_pic_num_of_each = 7;
test_pic_num_of_each = 3;
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
train_pic_num_of_each = 5;
test_pic_num_of_each = 2;
end

% 取出前30%作为测试数据，剩下70%作为训练数据
test_data_index = [];
train_data_index = [];
for i=0:people_num-1
    test_data_index = [test_data_index pic_num_of_each*i+1:pic_num_of_each*i+test_pic_num_of_each];
    train_data_index = [train_data_index pic_num_of_each*i+test_pic_num_of_each+1:pic_num_of_each*(i+1)];
end

train_data = reshaped_faces(:,train_data_index);
test_data = reshaped_faces(:, test_data_index);
dimension = row * column; %一张人脸的维度

% LDA
% 算每个类的平均
k = 1; 
class_mean = zeros(dimension, people_num); 
for i=1:people_num
    % 求一列（即一个人）的均值
    temp = class_mean(:,i);
    % 遍历每个人的train_pic_num_of_each张用于训练的脸，相加算平均
    for j=1:train_pic_num_of_each
        temp = temp + train_data(:,k);
        k = k + 1;
    end
    class_mean(:,i) = temp / train_pic_num_of_each;
end

% 算类类间散度矩阵Sb
Sb = zeros(dimension, dimension);
all_mean = mean(train_data, 2); % 全部的平均
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
    for j=1:train_pic_num_of_each % 遍历一个人的所有脸计算后相加
        centered_data = train_data(:,k) - class_mean(:,i);
        Sw = Sw + centered_data * centered_data';
        k = k + 1;
    end
end
Sw = Sw / (people_num * train_pic_num_of_each);

% 目标函数一：经典LDA（伪逆矩阵代替逆矩阵防止奇异值）
% target = pinv(Sw) * Sb;

% PCA
centered_face = (train_data - all_mean);
cov_matrix = centered_face * centered_face';
target = cov_matrix;

% 求特征值、特征向量
[eigen_vectors, dianogol_matrix] = eig(target);
eigen_values = diag(dianogol_matrix);

% 对特征值、特征向量进行排序
[sorted_eigen_values, index] = sort(eigen_values, 'descend'); 
eigen_vectors = eigen_vectors(:, index);
eigen_vectors = real(eigen_vectors); % 处理复数，会导致一定误差（LDA用）
rate = []; %用于记录人脸识别率
%使用SVM人脸识别
    % SVM(OVO)
for i=10:10:160
    right_num = 0;
    % 降维得到投影矩阵
    project_matrix = eigen_vectors(:,1:i);
    projected_train_data = project_matrix' * (train_data - all_mean);
    projected_test_data = project_matrix' * (test_data - all_mean);

    % SVM训练过程
           model_num = 1;
       for j = 0:1:people_num - 2
         train_img1 = projected_train_data(:,j * train_pic_num_of_each + 1 : j * train_pic_num_of_each + train_pic_num_of_each); % 取出每次SVM需要的训练集
         train_label1 = ones(1,train_pic_num_of_each)*(j + 1); % 给定训练标签
         test_img1 = projected_test_data(:,j * test_pic_num_of_each + 1 : j * test_pic_num_of_each + test_pic_num_of_each); % 取出每次SVM需要的测试集
         for z = j + 1:1:people_num - 1
         train_img2 = projected_train_data(:,z * train_pic_num_of_each + 1 : z * train_pic_num_of_each + train_pic_num_of_each); % 取出每次SVM需要的训练集
         train_label2 = ones(1,train_pic_num_of_each)*(z + 1); % 给定训练标签
         train_imgs = [train_img1,train_img2];
         train_label = [train_label1,train_label2];
         
         test_img2 = projected_test_data(:,z * test_pic_num_of_each + 1 : z * test_pic_num_of_each + test_pic_num_of_each); % 取出每次SVM需要的测试集
         test_imgs = [test_img1,test_img2];
         
          % 数据预处理,将训练集和测试集归一化到[0,1]区间 
        [mtrain,ntrain] = size(train_imgs); %m为行数，n为列数
        [mtest,ntest] = size(test_imgs);
 
        test_dataset = [train_imgs,test_imgs];
        % mapminmax为MATLAB自带的归一化函数
        [dataset_scale,ps] = mapminmax(test_dataset,0,1);
 
        train_imgs = dataset_scale(:,1:ntrain);
        test_imgs = dataset_scale( :,(ntrain+1):(ntrain+ntest) );
 
        % SVM网络训练
        train_imgs = train_imgs';
        train_label = train_label';
        expr = ['model_' num2str(model_num) ' = fitcsvm(train_imgs,train_label);']; % fitcsvm默认读取数据为按行，一张一脸为一列，需要转置
        eval(expr);
        model_num = model_num + 1;
         end
       end
       model_num = model_num - 1;
       
       % 人脸识别
       test = []; % 测试用
    for k = 1:1:test_pic_num_of_each * people_num
        test_img = projected_test_data(:,k); % 取出待识别图像
        test_real_label = fix((k - 1) / test_pic_num_of_each) + 1; % 给定待测试真实标签
        predict_labels = zeros(1,people_num); %用于OVO后续投票
      
       % SVM网络预测
       for t = 1:1:model_num
       predict_label = predict(eval(['model_' num2str(t)]),test_img');
       % test = [test,predict_label]; % 测试用
       predict_labels(1,predict_label) = predict_labels(1,predict_label) + 1;
       end
         [max_value,index] = max(predict_labels);
       if(index == test_real_label)
           right_num = right_num + 1;   
       end
    end
       
       recognition_rate = right_num / (test_pic_num_of_each * people_num); 
       rate = [rate,recognition_rate];
end          

%            % SVM(OVR)
% for i = 10:10:160
%         right_num = 0;
%     % 降维得到投影矩阵
%     project_matrix = eigen_vectors(:,1:i);
%     projected_train_data = project_matrix' * (train_data - all_mean);
%     projected_test_data = project_matrix' * (test_data - all_mean);
%          model_num = 1;
%              % SVM训练过程（每次训练都要使用整个数据集）
%          for j = 0:1:people_num - 1
%          
%          train_imgs = circshift(projected_train_data,-j * train_pic_num_of_each ,2); %使训练集始终在前几行
%          train_label1 = ones(1,train_pic_num_of_each) * (j + 1);
%          train_label2 = zeros(1,train_pic_num_of_each * (people_num - 1));
%          train_label = [train_label1,train_label2];
%          
%          test_imgs = circshift(projected_test_data,-j * test_pic_num_of_each ,2); %使测试集始终在前几行
% 
%         % 数据预处理,将训练集和测试集归一化到[0,1]区间 
%         [mtrain,ntrain] = size(train_imgs); %m为行数，n为列数
%         [mtest,ntest] = size(test_imgs);
%  
%         test_dataset = [train_imgs,test_imgs];
%         % mapminmax为MATLAB自带的归一化函数
%         [dataset_scale,ps] = mapminmax(test_dataset,0,1);
% 
%         train_imgs = dataset_scale(:,1:ntrain);
%         test_imgs = dataset_scale( :,(ntrain+1):(ntrain+ntest) );
%  
%         % SVM网络训练
%         train_imgs = train_imgs';
%         train_label = train_label';
%         expr = ['model_' num2str(model_num) ' = fitcsvm(train_imgs,train_label);']; % fitcsvm默认读取数据为按行，一张一脸为一列，需要转置
%         eval(expr);
%         model_num = model_num + 1;
%          end
%         model_num = model_num - 1;
%          % 人脸识别
%        for k = 1:1:test_pic_num_of_each * people_num
%         test_img = projected_test_data(:,k); % 取出待识别图像
%         test_real_label = fix((k - 1) / test_pic_num_of_each) + 1; % 给定待测试真实标签
%         predict_labels = zeros(1,people_num); %用于OVR预测
%       
%        % SVM网络预测
%        for t = 1:1:model_num
%        [predict_label,possibility] = predict(eval(['model_' num2str(t)]),test_img');
%        if predict_label ~= 0
%        predict_labels(1,predict_label) = predict_labels(1,predict_label) + possibility(1,1); 
%        end
%        end
%          [min_value,index] = min(predict_labels); % 若一张图片被预测为多类，选择离超平面最远的作为最终预测类
%        if(index == test_real_label)
%            right_num = right_num + 1;   
%        end
%        end
%        recognition_rate = right_num / (test_pic_num_of_each * people_num); 
%        rate = [rate,recognition_rate];
%         
% end

      
 

