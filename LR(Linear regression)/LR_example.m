x=[23.80,27.60,31.60,32.40,33.70,34.90,43.20,52.80,63.80,73.40];
y=[41.4,51.8,61.70,67.90,68.70,77.50,95.90,137.40,155.0,175.0];
figure
plot(x,y,'r*') %作散点图(制定横纵坐标)
xlabel('x(学生学习时间)','fontsize',12)
ylabel('y(学生成绩)','fontsize',12)
set(gca,'linewidth',2)
%采用最小二乘拟合
Lxx=sum((x-mean(x)).^2);
Lxy=sum((x-mean(x)).*(y-mean(y)));
b1=Lxy/Lxx;
b0=mean(y)-b1*mean(x);
y1=b1*x+b0; %线性方程用于预测和拟合
hold on
plot(x,y1,'linewidth',2);
m2=LinearModel.fit(x,y); %函数进行线性回归


