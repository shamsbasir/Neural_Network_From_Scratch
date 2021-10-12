% Empirical Part
% Q1)
% -epoch = 100
close all;clc;clear;
HiddenUnit       = [ 200 100 50 20 5];
AverageJ_train   = [ 0.0475 0.0454 0.0534 0.1357 0.5523];
AverageJ_test    = [0.4303 0.4405 0.4781 0.5727 0.7168];
figure(1)
plot(HiddenUnit,AverageJ_train,'linewidth',2.6);
hold on
plot(HiddenUnit,AverageJ_test,'linewidth',2.6);
xticks([5 20 50 100 200]);
xlabel("Number of hidden units");
ylabel("Average Cross Entropy");
legend({"training data","test data"},'fontsize',16);
set(gca,'FontSize',16)
saveas(gcf,'Q1.png')


% Q3) : -epoch = 100
% lr = 0.1
%a = load("train_01.csv",'-ascii');
%b = load("test_01.csv",'-ascii');
load('a');
load('b');
training_J = a(:,2);
test_J     = b(:,2);
epoch      = a(:,1);
figure(2)
plot(epoch,training_J,'linewidth',2.6);
hold on
plot(epoch,test_J,'linewidth',2.6);
xlabel("Number of epochs");
ylabel("Average Cross Entropy");
legend("training data","test data");
title("lr = 0.1");
set(gca,'FontSize',16)
saveas(gcf,'lr0_1.png')


% lr = 0.01
tr = load("train_001.csv",'-ascii');
ts = load("test_001.csv",'-ascii');
training_J = tr(:,2);
test_J     = ts(:,2);
epoch      = tr(:,1);
figure(3)
plot(epoch,training_J,'linewidth',2.6);
hold on
plot(epoch,test_J,'linewidth',2.6);
xlabel("Number of epochs");
ylabel("Average Cross Entropy");
legend("training data","test data");
title("lr = 0.01");
set(gca,'FontSize',16)
saveas(gcf,'lr0_01.png')

%lr = 0.001
tr = load("train_0001.csv",'-ascii');
ts = load("test_0001.csv",'-ascii');
training_J = tr(:,2);
test_J     = ts(:,2);
epoch      = tr(:,1);
figure(4)
plot(epoch,training_J,'linewidth',2.6);
hold on
plot(epoch,test_J,'linewidth',2.6);
xlabel("Number of epochs");
ylabel("Average Cross Entropy");
legend("training data","test data");
title("lr = 0.001");
set(gca,'FontSize',16)
saveas(gcf,'lr0_001.png')