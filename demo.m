%% demo for running the algotirhms: BMCH-ST and BMCG-GH
clc;clear;
close all;
rng(1,'twister');
%% data preprocess
load 'SSF_100m.mat'
 Z_true = SSF;

[m,n]=size(Z_true);
[X,Y_sample] = meshgrid(0:n-1,0:m-1);
Z = Z_true;

% add noises
sigma2 = 1; 
NOISE = sqrt(sigma2)*randn(m,n);
Z = Z + NOISE;

%% sample the field
p = 0.2; % sapmling rate
pmn = round(p*m*n);
Omega = randperm(m*n); Omega = Omega(1:pmn);
P = zeros(m,n);  P(Omega) = 1;%create observation matrix


%% plot the field
figure
subplot(2,3,1)
pcolor( Z_true);
        shading interp;
        colormap('jet');
%         colorbar;
        axis image
title('original field')

subplot(2,3,2);
a = P.*Z;
a(a==0)=nan;
imagesc(X(1,:),Y_sample(:,1),flipud(a));
axis image
colormap('jet');
title('sampled field')

%% proprocess
f = Z_true;
Y=P.*Z;
mean_Y = sum(Y(:))/(numel(find(Y~=0)));
for i=1:m
    for j=1:n
        if(Y(i,j)~=0)
            Y(i,j) = Y(i,j) - mean_Y;
        end
    end
end

%% BMCG-ST
[m,n]=size(f);
Lu1 = zeros(m); 
theta = sqrt(3);

for ii = 1 : m
    for jj = 1 : m
        Lu1(ii,jj) = exp(-(ii-jj)^2/theta^2);
    end
end
Lu = diag(sum(Lu1)) - Lu1 + eye(m)*(1e-6);
Lv1 = zeros(n); 
for ii = 1 : n
    for jj = 1 : n
        Lv1(ii,jj) = exp(-(ii-jj)^2/theta^2);
    end
end
Lv = diag(sum(Lv1)) - Lv1 + eye(n)*(1e-6);

[Result, stat] = BMCG_ST(Y, 'obs', P, 'initRank',round(min(m,n)/2),...
    'maxiters', 100, 'Gu', Lu, 'Gv', Lv);
Result.X = Result.X + mean_Y;

subplot(2,3,3);
pcolor( Result.X  )
        shading interp;
        colormap('jet');
        axis image
title('BMCG');
a = abs((Result.X-Z_true));
MAE_BMCG = sum(a(:))/(m*n);
RMSE_BMCG = norm((Result.X-Z_true),'fro')/sqrt(m*n);

%% BMCG-ST(M)
range = 16;
mag = 1e2;
func = @(h)mag*(1+sqrt(5)*h/range+5*h.^2/(3*range.^2))*exp(-sqrt(5)*h./(range));%matern 2.5
K = zeros(m,n);
for i=1:m
    for j=1:n
        K(i,j)=func(abs(i-j));
    end
end
K = inv(K+ eye(m,n)*(1e-6));

[Result2, stat2] = BMCG_ST(Y, 'obs', P, 'initRank',round(min(m,n)/2),...
    'maxiters', 100, 'Gu', K, 'Gv', K);
Result2.X = Result2.X + mean_Y;

subplot(2,3,4);
pcolor( Result2.X  )
        shading interp;
        colormap('jet');
        axis image
title('BMCG(M)');

a = abs((Result2.X-Z_true));
MAE_BMCG_m = sum(a(:))/(m*n);
RMSE_BMCG_m = norm((Result2.X-Z_true),'fro')/sqrt(m*n);

%% BMCG_GH
[Result_BMCG_GH, stat_BMCG_GH] = BMCG_GH(Y,'init','ml', 'obs', P, 'initRank',round(min(m,n)/2),...
    'maxiters', 150, 'Gu', Lu, 'Gv', Lv,'dimRed',1);
Result_BMCG_GH.X = Result_BMCG_GH.X + mean_Y;

subplot(2,3,5);
pcolor( Result_BMCG_GH.X  )
        shading interp;
        colormap('jet');
        axis image
title('BMCG-GH');

err_BMCG_GH = norm(Result_BMCG_GH.X-f,'fro')/norm(f,'fro');
a = abs((Result_BMCG_GH.X-Z_true));
MAE_BMCG_GH = sum(a(:))/(m*n);
RMSE_BMCG_GH = norm((Result_BMCG_GH.X-Z_true),'fro')/sqrt(m*n);

%% BMCG_GH(M)
[Result_BMCG_GH_M, stat_BMCG_GH_M] = BMCG_GH(Y,'init','rand', 'obs', P, 'initRank',round(min(m,n)/2),...
    'maxiters',150, 'Gu', K, 'Gv', K,'dimRed',1);
Result_BMCG_GH_M.X = Result_BMCG_GH_M.X + mean_Y;

subplot(2,3,6);
pcolor( Result_BMCG_GH_M.X  )
        shading interp;
        colormap('jet');
        axis image
title('BMCG-GH(M)');

err_BMCG_GH_M = norm(Result_BMCG_GH_M.X-f,'fro')/norm(f,'fro');
a = abs((Result_BMCG_GH_M.X-Z_true));
MAE_BMCG_GH_M = sum(a(:))/(m*n);
RMSE_BMCG_GH_M = norm((Result_BMCG_GH_M.X-Z_true),'fro')/sqrt(m*n);
