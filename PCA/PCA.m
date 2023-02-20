clear;
% 1.人脸数据集的导入与数据处理（400张图，一共40人，一人10张）
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
reshaped_faces=[];
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

% 取出前30%作为测试数据，剩下70%作为训练数据
test_data_index = [];
train_data_index = [];
for i=0:39
    test_data_index = [test_data_index 10*i+1:10*i+3];
    train_data_index = [train_data_index 10*i+4:10*(i+1)];
end

% %对照组（测试与训练比例不变，取的图像有变化）
% % 取出前70%作为训练数据，剩下30%作为测试数据
% test_data_index = [];
% train_data_index = [];
% for i=0:39
%     train_data_index = [train_data_index 10*i+1:10*i+7];
%     test_data_index = [test_data_index 10*i+8:10*(i+1)];
% end

train_data = reshaped_faces(:,train_data_index);
test_data = reshaped_faces(:, test_data_index);

%用于展示重塑后某些训练图片 测试用
%show_faces(train_data); 

% 2.图像求均值，中心化
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mean_face = mean(train_data,2);
%waitfor(show_face(mean_face)); %平均脸展示，测试用

centered_face = (train_data - mean_face);
%waitfor(show_faces(centered_face)); %用于展示中心化后某些训练图片 测试用

% 3.求协方差矩阵、特征值与特征向量并排序
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cov_matrix = centered_face * centered_face';
[eigen_vectors, dianogol_matrix] = eig(cov_matrix);

% 从对角矩阵获取特征值
eigen_values = diag(dianogol_matrix);

% 对特征值按索引进行从大到小排序
[sorted_eigen_values, index] = sort(eigen_values, 'descend'); 

% 获取排序后的征值对应的特征向量
sorted_eigen_vectors = eigen_vectors(:, index);

all_eigen_faces = sorted_eigen_vectors;

%waitfor(show_faces(all_eigen_faces)); %用于展示某些特征脸 测试用

% 4.人脸重构
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 取出第一个人的人脸（随意换），用于重建
single_face = centered_face(:,1);

index = 1;
for dimensionality=20:20:160

    % 取出相应数量特征脸（前n大特征值对应的特征向量，用于重构人脸）
    eigen_faces = all_eigen_faces(:,1:dimensionality);

    % 重建人脸并显示
        rebuild_face = eigen_faces * (eigen_faces' * single_face) + mean_face;
        subplot(2, 4, index); %两行四列
        index = index + 1;
        fig = show_face(rebuild_face);
        title(sprintf("dimensionality=%d", dimensionality));    
        if (dimensionality == 160)
            waitfor(fig);
        end
end

% 5.人脸识别
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

index = 1;       
Y = [];
% KNN
for k=1:6

    for i=10:10:160
   eigen_faces = all_eigen_faces(:,1:i);
    % 测试、训练数据降维
    projected_train_data = eigen_faces' * (train_data - mean_face);
    projected_test_data = eigen_faces' * (test_data - mean_face);
        % 用于保存最小的k个值的矩阵
        % 用于保存最小k个值对应的人标签的矩阵
        minimun_k_values = zeros(k,1);
        label_of_minimun_k_values = zeros(k,1);

        % 测试脸的数量
        test_face_number = size(projected_test_data, 2);

        % 识别正确数量
        correct_predict_number = 0;

        % 遍历每一个待测试人脸
        for each_test_face_index = 1:test_face_number

            each_test_face = projected_test_data(:,each_test_face_index);

            % 先把k个值填满，避免在迭代中反复判断
            for each_train_face_index = 1:k
                minimun_k_values(each_train_face_index,1) = norm(each_test_face - projected_train_data(:,each_train_face_index));
                label_of_minimun_k_values(each_train_face_index,1) = floor((train_data_index(1,each_train_face_index) - 1) / 10) + 1;
            end

            % 找出k个值中最大值及其下标
            [max_value, index_of_max_value] = max(minimun_k_values);

            % 计算与剩余每一个已知人脸的距离
            for each_train_face_index = k+1:size(projected_train_data,2)

                % 计算距离
                distance = norm(each_test_face - projected_train_data(:,each_train_face_index));

                % 遇到更小的距离就更新距离和标签
                if (distance < max_value)
                    minimun_k_values(index_of_max_value,1) = distance;
                    label_of_minimun_k_values(index_of_max_value,1) = floor((train_data_index(1,each_train_face_index) - 1) / 10) + 1;
                    [max_value, index_of_max_value] = max(minimun_k_values);
                end
            end

            % 最终得到距离最小的k个值以及对应的标签
            % 取出出现次数最多的值，为预测的人脸标签
            predict_label = mode(label_of_minimun_k_values);
            real_label = floor((test_data_index(1,each_test_face_index) - 1) / 10)+1;

            if (predict_label == real_label)
                correct_predict_number = correct_predict_number + 1;
            end
        end
        % 单次识别率
        correct_rate = correct_predict_number/test_face_number;
        Y = [Y correct_rate];
    end
end

% 求不同k值不同维度下的人脸识别率及对k的平均识别率
Y = reshape(Y,k,16);
waitfor(waterfall(Y));
avg_correct_rate=mean(Y);
waitfor(plot(avg_correct_rate));

% 6.人脸图像降维 数据二三维可视化（可推广到不同数据集）
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i=[2 3]

    eigen_faces = all_eigen_faces(:,1:i);

    % 投影
    projected_test_data = eigen_faces' * (test_data - mean_face);

    color = [];
    for j=1:120
        color = [color floor((j-1)/4)*5];
    end

    if (i == 2)
        waitfor(scatter(projected_test_data(1, :), projected_test_data(2, :), [], color));
    else
        waitfor(scatter3(projected_test_data(1, :), projected_test_data(2, :), projected_test_data(3, :), [], color));
    end

end

%内用函数定义
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 输入向量，显示脸
function fig = show_face(vector)
    fig = imshow(mat2gray(reshape(vector, [50, 40])));
end

% 显示矩阵中某些脸
function fig = show_faces(eigen_vectors)
    count = 1;
    index_of_image_to_show = [1,5,10,15,20,30,50,70,100,150];
    for i=index_of_image_to_show
        subplot(2,5,count);
        fig = show_face(eigen_vectors(:, i));
        title(sprintf("i=%d", i));
        count = count + 1;
    end
end