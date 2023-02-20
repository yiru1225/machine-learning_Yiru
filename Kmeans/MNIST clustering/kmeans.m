time_while = 1; %设置循环次数
rate = [1,time_while];
for time = 1:time_while
k = 5; % 设置k
right_num = 0; %正确个数
num_sample = 1000; %待聚类个数
train = reshape(train_images,784,60000);
train_data = train(:,1:num_sample); %取前n列做训练
% K-means训练

%[idx_pc,center_pc] = kmeans(train_data',k); %idx是分类类别，center是质心集

data = train_data;
times = 0;
N = k;
%%  初始化工作
[n,m] = size(data); % m = 列,n = 行
center = zeros(n,N);% 初始化聚类中心，生成n行N列的零矩阵
pattern = data;     % 将整个数据拷贝到pattern矩阵中
%% 算法
for x = 1 : N
    % 第一次随机产生聚类中心 randperm随机取数
    center(:,x) = data(:,randperm(num_sample,1)); 
end
while true
distence = zeros(1,N);   % 产生1行N列的零矩阵
num = zeros(1,N);        % 产生1行N列的零矩阵
new_center = zeros(n,N); % 产生n行N列的零矩阵
%% 将所有的点打上标签1 2 3...N
for x = 1 : m
    for y = 1 : N
        distence(y) = norm(data(:,x) - center(:,y)); % norm函数计算到每个类的距离
    end
    [~,temp] = min(distence); %求最小的距离 ~是距离值，temp是第几个
    pattern(n + 1,x) = temp;         
end
times = times+1;
tag = 0;
%% 将所有在同一类里的点坐标全部相加，计算新的中心坐标
for y = 1 : N
    for x = 1 : m
        if pattern(n + 1,x) == y
           new_center(:,y) = new_center(:,y) + pattern(1:n,x);
           num(y) = num(y) + 1;
        end
    end
    new_center(:,y) = new_center(:,y) / num(y);
    if norm(new_center(:,y) - center(:,y)) > 0.0001 %设定最小误差变化（阈值）
        tag = 1;
    end
end
if tag == 0 || times > 10000 % 设置终止条件（加入最大迭代次数限制）
     break;
else
     center = new_center;
end
end

% 聚类正确率
 idx = pattern(785:785,:);
%idx = idx_pc'; % 调库正确率分析
for i = 1:k
    num_i = 0;
    index = zeros(1,num_sample);
    index_label = zeros(1,num_sample);
    for j = 1:num_sample
      if idx(1,j) == i
         num_i = num_i + 1;
         index(num_i) = j;
         index_label(num_i) = train_labels(:,j) + 1;
      end    
    end
    %找出第二多出现的数
    test = mode(index_label);
      index_label(index_label==test) = [];
      [test2,n] = mode(index_label);
        right_num = right_num + n;
end
rate(time) = right_num / num_sample;
end
 plot(rate)
% %%PCA降维可视化
% % 2.图像求均值，中心化
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mean_data = mean(data,2);
% centered_data = (data - mean_data);
% % 3.求协方差矩阵、特征值与特征向量并排序
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% cov_matrix = centered_data * centered_data';
% [eigen_vectors, dianogol_matrix] = eig(cov_matrix);
% 
% % 从对角矩阵获取特征值
% eigen_values = diag(dianogol_matrix);
% 
% % 对特征值按索引进行从大到小排序
% [sorted_eigen_values, index] = sort(eigen_values, 'descend'); 
% 
% % 获取排序后的征值对应的特征向量
% sorted_eigen_vectors = eigen_vectors(:, index);
% 
% all_eigen_data = sorted_eigen_vectors;
% 
% %可视化
% 
% eigen_data = all_eigen_data(:,1:i);
% i = 3; %降维数
%     % 投影
%     projected_test_data = eigen_data' * (data - mean_data);
% 
%     color = [];
%     for j=1:num_sample
%         color = [color floor((j-1)/4)*5];
%     end
% 
%     if (i == 2)
%         waitfor(scatter(projected_test_data(1, :), projected_test_data(2, :), [], color));
%     else
%         waitfor(scatter3(projected_test_data(1, :), projected_test_data(2, :), projected_test_data(3, :), [], color));
%     end

% %%LDA降维可视化
% % 算每个类的平均
% k = 1; 
% class_mean = zeros(dimension, people_num); 
% for i=1:people_num
%     % 求一列（即一个人）的均值
%     temp = class_mean(:,i);
%     % 遍历每个人的train_pic_num_of_each张用于训练的脸，相加算平均
%     for j=1:train_pic_num_of_each
%         temp = temp + train_data(:,k);
%         k = k + 1;
%     end
%     class_mean(:,i) = temp / train_pic_num_of_each;
% end
% 
% % 算类类间散度矩阵Sb
% Sb = zeros(dimension, dimension);
% all_mean = mean(train_data, 2); % 全部的平均
% for i=1:people_num
%     % 以每个人的平均脸进行计算，这里减去所有平均，中心化
%     centered_data = class_mean(:,i) - all_mean;
%     Sb = Sb + centered_data * centered_data';
% end
% Sb = Sb / people_num;
% 
% % 算类内散度矩阵Sw
% Sw = zeros(dimension, dimension);
% k = 1; % p表示每一张图片
% for i=1:people_num % 遍历每一个人
%     for j=1:train_pic_num_of_each % 遍历一个人的所有脸计算后相加
%         centered_data = train_data(:,k) - class_mean(:,i);
%         Sw = Sw + centered_data * centered_data';
%         k = k + 1;
%     end
% end
% Sw = Sw / (people_num * train_pic_num_of_each);
% 
% % 目标函数一：经典LDA（伪逆矩阵代替逆矩阵防止奇异值）
% % target = pinv(Sw) * Sb;
% 
% % 目标函数二：不可逆时需要正则项扰动
% %   Sw = Sw + eye(dimension)*10^-6;
% %   target = Sw^-1 * Sb;
% 
% % 目标函数三：相减形式
% % target = Sb - Sw;
% 
% % 目标函数四：相除
% % target = Sb/Sw;
% 
% % 目标函数五：调换位置
% % target = Sb * pinv(Sw);








