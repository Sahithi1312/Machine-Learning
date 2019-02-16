clc;
close all;
clear;

C1 = [1.7 3.0; 1.2 3.5; 1.5 5.0; 0.6 4.5; 0.5 4.0];
C2 = [1.1 -1.3; 0.2 3.5; 1.5 4.3; 3.2 0.8; 4.0 2.7];

mu_1 = mean(C1);
fprintf('\n Mean of Class 1 = [%f %f] \n', mu_1);

mu_2 = mean(C2);
fprintf('\n Mean of Class 2 = [%f %f] \n', mu_2);

sigma_1 = cov(C1);
fprintf('\n Sigma of Class 1 = [%f %f \n\t\t\t\t\t %f %f] \n', sigma_1);

sigma_2 = cov(C2);
fprintf('\n Sigma of Class 2 = [%f %f \n\t\t\t\t\t %f %f] \n', sigma_2);

mle_mu_1 = mle(mu_1);
fprintf('\n Unbiased Maximum Likelihood of mean (C1) = [%f %f]\n', mle_mu_1);

mle_mu_2 = mle(mu_2);
fprintf('\n Unbiased Maximum Likelihood of mean (C2)= [%f %f]\n', mle_mu_2);

mle_sigma_1_row1 = mle(sigma_1(1,:));
mle_sigma_1_row2 = mle(sigma_1(2,:));
fprintf('\n Unbiased Maximum Likelihood of sigma (C1) = \n\n\n');
disp(mle_sigma_1_row1);
disp(mle_sigma_1_row2);

mle_sigma_2_row1 = mle(sigma_2(1,:));
mle_sigma_2_row2 = mle(sigma_2(2,:));
fprintf('\n Unbiased Maximum Likelihood of sigma (C2) = \n\n\n');
disp(mle_sigma_2_row1);
disp(mle_sigma_2_row2);

fprintf('\n');

data_points_C1 = mvnrnd(mu_1,sigma_1, 10);
data_points_C2 = mvnrnd(mu_2,sigma_2, 10);

subplot(1,2,1)
plot(data_points_C1(:,1),data_points_C1(:,2),'x')
hold on
plot(data_points_C2(:,1),data_points_C2(:,2),'o')
legend( 'x -> Class 1','o -> Class 2', 'Location', 'best');
title('Class 1 and Class 2')

data_points_C3 = mvnrnd([11 40],sigma_1, 10);
subplot(1,2,2)
plot(data_points_C1(:,1),data_points_C1(:,2),'x', 'DisplayName', 'x -> Class 1')
hold on
plot(data_points_C3(:,1),data_points_C3(:,2),'o', 'DisplayName', 'o -> Class 3')
legend( 'x -> Class 1','o -> Class 3', 'Location', 'southeast');
title('Class 1 and Class 3')

