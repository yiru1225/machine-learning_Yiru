clear;
clc;
times = 0;
N = input('请设置聚类数目：');%设置聚类数目
%% 第一组数据
mu1=[0 0];  %均值
S1=[0.1 0 ; 0 0.1];  %协方差
data1=mvnrnd(mu1,S1,10);   %产生高斯分布数据
%% 第二组数据
mu2=[-1.25 1.25];
S2=[0.1 0 ; 0 0.1];
data2=mvnrnd(mu2,S2,10);
%% 第三组数据
mu3=[1.25 1.25];
S3=[0.1 0 ; 0 0.1];
data3=mvnrnd(mu3,S3,10);
%% 显示数据
plot(data1(:,1),data1(:,2),'b+');
hold on;
plot(data2(:,1),data2(:,2),'b+');
plot(data3(:,1),data3(:,2),'b+');
%%  初始化工作
data = [data1;data2;data3];
[m,n] = size(data); % m = 30,n = 2
center = zeros(N,n);% 初始化聚类中心，生成N行n列的零矩阵
pattern = data;     % 将整个数据拷贝到pattern矩阵中
%% 算法
for x = 1 : N
    center(x,:) = data(randi(3,1),:); % 第一次随机产生聚类中心 randi返回1*1的(1,300)的数
end
while true
distence = zeros(1,N);   % 产生1行N列的零矩阵
num = zeros(1,N);        % 产生1行N列的零矩阵
new_center = zeros(N,n); % 产生N行n列的零矩阵
%% 将所有的点打上标签1 2 3...N
for x = 1 : m
    for y = 1 : N
        distence(y) = norm(data(x,:) - center(y,:)); % norm函数计算到每个类的距离
    end
    [~,temp] = min(distence); %求最小的距离 ~是距离值，temp是第几个
    pattern(x,n + 1) = temp;         
end
times = times+1;
k = 0;
%% 将所有在同一类里的点坐标全部相加，计算新的中心坐标
for y = 1 : N
    for x = 1 : m
        if pattern(x,n + 1) == y
           new_center(y,:) = new_center(y,:) + pattern(x,1:n);
           num(y) = num(y) + 1;
        end
    end
    new_center(y,:) = new_center(y,:) / num(y);
    if norm(new_center(y,:) - center(y,:)) < 0.0001 %设定最小误差变化（阈值）
        k = k + 1;
    end
end
if k == N || times > 10000 % 设置终止条件（加入最大迭代次数限制）
     break;
else
     center = new_center;
end
end
[m, n] = size(pattern); %[m,n] = [30,3]
 
%% 最后显示聚类后的数据
figure;
hold on;
for i = 1 : m
    if pattern(i,n) == 1 
         plot(pattern(i,1),pattern(i,2),'r*');
         plot(center(1,1),center(1,2),'ko');
    elseif pattern(i,n) == 2
         plot(pattern(i,1),pattern(i,2),'g*');
         plot(center(2,1),center(2,2),'ko');
    elseif pattern(i,n) == 3
         plot(pattern(i,1),pattern(i,2),'b*');
         plot(center(3,1),center(3,2),'ko');
    elseif pattern(i,n) == 4
         plot(pattern(i,1),pattern(i,2),'y*');
         plot(center(4,1),center(4,2),'ko');
    else
         plot(pattern(i,1),pattern(i,2),'m*');
         plot(center(5,1),center(5,2),'ko');
    end
end
