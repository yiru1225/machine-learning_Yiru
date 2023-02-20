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
train_pic_num_of_each = 7;
test_pic_num_of_each = 3;
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
dimension = row * column;

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

% 目标函数二：不可逆时需要正则项扰动
%   Sw = Sw + eye(dimension)*10^-6;
%   target = Sw^-1 * Sb;

% 目标函数三：相减形式
% target = Sb - Sw;

% 目标函数四：相除
% target = Sb/Sw;

% 目标函数五：调换位置
% target = Sb * pinv(Sw);

%PCA
% centered_face = (train_data - all_mean);
% cov_matrix = centered_face * centered_face';
% target = cov_matrix;

% 求特征值、特征向量
[eigen_vectors, dianogol_matrix] = eig(target);
eigen_values = diag(dianogol_matrix);

% 对特征值、特征向量进行排序
[sorted_eigen_values, index] = sort(eigen_values, 'descend'); 
eigen_vectors = eigen_vectors(:, index);

% 人脸识别

index = 1;
X = [];
Y = [];
% i为降维维度
for i=1:5:101

    % 投影矩阵
    project_matrix = eigen_vectors(:,1:i);
    projected_train_data = project_matrix' * (train_data - all_mean);
    projected_test_data = project_matrix' * (test_data - all_mean);

    % KNN的k值
    K=1;

    % 用于保存最小的k个值的矩阵
    % 用于保存最小k个值对应的人标签的矩阵
    minimun_k_values = zeros(K,1);
    label_of_minimun_k_values = zeros(K,1);

    % 测试脸的数量
    test_face_number = size(projected_test_data, 2);

    % 识别正确数量
    correct_predict_number = 0;

    % 遍历每一个待测试人脸 
    for each_test_face_index = 1:test_face_number

        each_test_face = projected_test_data(:,each_test_face_index);

        % 先把k个值填满，避免在迭代中反复判断
        for each_train_face_index = 1:K
            minimun_k_values(each_train_face_index,1) = norm(each_test_face - projected_train_data(:,each_train_face_index));
            label_of_minimun_k_values(each_train_face_index,1) = floor((train_data_index(1,each_train_face_index) - 1) / pic_num_of_each) + 1;
        end

        % 找出k个值中最大值及其下标
        [max_value, index_of_max_value] = max(minimun_k_values);

        % 计算与剩余每一个已知人脸的距离
        for each_train_face_index = K+1:size(projected_train_data,2)

            % 计算距离
            distance = norm(each_test_face - projected_train_data(:,each_train_face_index));

            % 遇到更小的距离就更新距离和标签
            if (distance < max_value)
                minimun_k_values(index_of_max_value,1) = distance;
                label_of_minimun_k_values(index_of_max_value,1) = floor((train_data_index(1,each_train_face_index) - 1) / pic_num_of_each) + 1;
                [max_value, index_of_max_value] = max(minimun_k_values);
            end
        end

        % 最终得到距离最小的k个值以及对应的标签
        % 取出出现次数最多的值，为预测的人脸标签
        predict_label = mode(label_of_minimun_k_values);
        real_label = floor((test_data_index(1,each_test_face_index) - 1) / pic_num_of_each)+1;

        if (predict_label == real_label)
            correct_predict_number = correct_predict_number + 1;
        end
    end

    correct_rate = correct_predict_number/test_face_number;

    X = [X i];
    Y = [Y correct_rate];

    fprintf("k=%d，i=%d，总测试样本：%d，正确数:%d，正确率：%1f\n", K, i,test_face_number,correct_predict_number,correct_rate);

    if (i == 101)
        waitfor(plot(X,Y));
    end
end

% 二三维可视化
class_num_to_show = 3;
pic_num_in_a_class = pic_num_of_each;
pic_to_show = class_num_to_show * pic_num_in_a_class;
for i=[2 3]

    % 取出相应数量特征向量
    project_matrix = eigen_vectors(:,1:i);

    % 投影
    projected_test_data = project_matrix' * (reshaped_faces - all_mean);
    projected_test_data = projected_test_data(:,1:pic_to_show);

    color = [];
    for j=1:pic_to_show
        color = [color floor((j-1)/pic_num_in_a_class)*20];
    end

    % 显示
    if (i == 2)
        subplot(1, 7, [1, 2, 3, 4]);
        scatter(projected_test_data(1, :), projected_test_data(2, :), [], color, 'filled');
        for j=1:3
            subplot(1, 7, j+4);
            fig = show_face(test_data(:,floor((j - 1) * pic_num_in_a_class) + 1), row, column);
        end
        waitfor(fig);
    else
        subplot(1, 7, [1, 2, 3, 4]);
        scatter3(projected_test_data(1, :), projected_test_data(2, :), projected_test_data(3, :), [], color, 'filled');
        for j=1:3
            subplot(1, 7, j+4);
            fig = show_face(test_data(:,floor((j - 1) * pic_num_in_a_class) + 1), row, column);
        end
        waitfor(fig);
    end
end

% 输入向量，显示脸
function fig = show_face(vector, row, column)
    fig = imshow(mat2gray(reshape(vector, [row, column])));
end
