import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import random
import time
from sklearn.decomposition import PCA


def display(X, C):
    color = ['blue', 'red', 'green', 'orange', 'pink', 'gray', 'olive', 'cyan', 'pink', 'purple', 'brown']
    pca = PCA(n_components=2)
    tmp_data = X
    # X_2d = pca.components_.T
    pca.fit(tmp_data)
    C_list = []
    for c_list in C:
        tmp_c = []
        for ci in c_list:
            tmp_c.append(ci.ravel())
        C_list.append(tmp_c)
    X_new = []
    for C in C_list:
        X_new.append(pca.transform(C))
    # show src_data
    for i in range(len(X_new)):
        for x in X_new[i]:
            plt.scatter(x[0], x[1], c=color[labels[0][i]], marker='o', s=15)
    plt.show()
    # show dst_data
    for i in range(len(X_new)):
        for x in X_new[i]:
            plt.scatter(x[0], x[1], c=color[i], marker='o', s=15)

    plt.show()


if __name__ == '__main__':

    images_file = 'C:/Users/hp/Desktop/最优化方法/实验一/train_images.mat'
    labels_file = 'C:/Users/hp/Desktop/最优化方法/实验一/train_labels.mat'
    images = scio.loadmat(images_file)
    images = images['train_images']  # size:(28,28,60000)
    labels = scio.loadmat(labels_file)
    labels = labels['train_labels']  # size:(1,60000)

    total_time = 0
    total_rate = 0
    times = 5  # 重复进行的次数
    rate_set = []

    for j in range(times):
        counter = 1
        start = time.time()
        m = 1000  # 前m张图片
        k = 10  # 类别数
        index = [i for i in range(m)]
        miu = []  # 所有质心
        X = []
        for i in range(m):
            xi = images[:, :, i]
            X.append(xi.ravel())
        X = np.array(X)
        init_index = random.sample(index, k)
        for i in init_index:  # 随机选择样本作为质心
            miu.append(images[:, :, i])
        num_iters = 0 #迭代次数
        while True:
            C = []  # 所有的类
            C_labels = []
            for i in range(k):
                Ci = []  # 每一类
                Ci_labels = []
                C.append(Ci)
                C_labels.append(Ci_labels)
            # 将样本划入最近的质心对应的类
            for i in range(m):
                xi = images[:, :, i]
                # F2范数计算距离，即欧几里得距离
                # 每个点都看看和哪一个质心距离最近
                d = [np.linalg.norm(xi - miu[j]) for j in range(k)]
                min_i = d.index(min(d))  # 质心距离最小的下标
                # 这个质心吸收了这个点
                C[min_i].append(xi)
                C_labels[min_i].append(labels[0, i])
            # 更新质心
            update = False
            for i in range(k):
                xi = np.zeros((28, 28), dtype='float32')
                # 重新计算质心
                for xj in C[i]:
                    xi += xj
                xi /= len(C[i])
                # 质心改变，则更新
                if not (miu[i] == xi).all():
                    update = True
                miu[i] = xi
            # 已经迭代了20次
            num_iters += 1
            if num_iters == 20:
                break
            if not update:
                break
            correct = 0
            # 精度计算
            # C_labels 中有10个 list
            # print(len(C_labels))
            # 每个list中有正确的label，选择最多的label作为该类的label，计算该类label有多少个
            c_belong = 1
            for label_list in C_labels:
                correct_each = 0
                label = max(label_list, key=label_list.count)  # 计数 拿最多
                print("%d---%d" % (c_belong, label))
                c_belong += 1
                for t in label_list:
                    # 计数list中label与计数最多的相同的个数
                    if t == label:
                        correct_each += 1
                correct += correct_each
            print('try %d times get accurate %.2f%%' % (num_iters, 100 * correct / m))
        # print('___________________')

        end = time.time()
        total_time += end - start
        print('%.3fs  ' % (end - start), end="")

        # 计算聚类精度
        correct = 0
        for label_list in C_labels:
            correct_each = 0
            label = max(label_list, key=label_list.count)
            for t in label_list:
                if t == label:
                    correct_each += 1
            correct += correct_each

        rate_set.append(correct / m)
        print('%.2f%%' % (100 * correct / m))

        #total_rate += correct / m

    print(rate_set)
    max_indx = np.max(rate_set)  # max value index
    min_indx = np.min(rate_set)  # min value index
    mean_indx = np.mean(rate_set)  # mean
    var_indx = np.var(rate_set) #var
    print(max_indx, min_indx, mean_indx,var_indx)

    plt.figure()
    plt.plot(np.arange(times) , rate_set, color="black",lw=1.5)
    plt.grid(True)
    plt.title("K-means Accuracy ")
    plt.xlabel("Number of times")
    plt.ylabel("Accuracy")
    plt.show()

    display(X, C)
    print()
    print('前%d张' % m)
    print('%.3fs  ' % (total_time / times), end="")
    # print('%.2f%%' % (100 * total_rate / times))

