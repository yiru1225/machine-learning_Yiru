clear;
% dataset是将bedroom和forest合并；dataset = [bedroom;forset];这行代码可以实现合并
load forest.mat                  %导入要分类的数据集
load bedroom.mat
dataset = [bedroom;MITforest];
load labelset.mat                %导入分类集标签集

% 选定训练集和测试集

% 将第一类的1-5,第二类的11-15做为训练集
train_set = [dataset(1:5,:);dataset(11:15,:)];
% 相应的训练集的标签也要分离出来
train_set_labels = [lableset(1:5);lableset(11:15)];
% 将第一类的6-10,第二类的16-20,做为测试集
test_set = [dataset(6:10,:);dataset(16:20,:)];
% 相应的测试集的标签也要分离出来
test_set_labels = [lableset(6:10);lableset(16:20)];
 
% 数据预处理,将训练集和测试集归一化到[0,1]区间
 
[mtrain,ntrain] = size(train_set);
[mtest,ntest] = size(test_set);
 
test_dataset = [train_set;test_set];
% mapminmax为MATLAB自带的归一化函数
[dataset_scale,ps] = mapminmax(test_dataset',0,1);
dataset_scale = dataset_scale';
 
train_set = dataset_scale(1:mtrain,:);
test_set = dataset_scale( (mtrain+1):(mtrain+mtest),: );
 
% SVM网络训练
model = fitcsvm(train_set,train_set_labels);
 
% SVM网络预测
[predict_label] = predict(model,test_set);
%[predict_label] = model.IsSupportVector;

 
% 结果分析
 
% 测试集的实际分类和预测分类图
% 通过图可以看出只有一个测试样本是被错分的
figure;
hold on;
plot(test_set_labels,'o');
plot(predict_label,'r*');
xlabel('测试集样本','FontSize',12);
ylabel('类别标签','FontSize',12);
legend('实际测试集分类','预测测试集分类');
title('测试集的实际分类和预测分类图','FontSize',12);
grid on;
