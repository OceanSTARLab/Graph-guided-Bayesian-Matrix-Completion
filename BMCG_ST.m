function[Result, stat] = BMCG_ST(M, varargin)

%% Set parameters
[m,n] = size(M);

ip = inputParser;
ip.addParameter('obs', ones(m,n),@(x)(isnumeric(x)||islogical(x)));
% ip.addParameter('V', ones(m,n),@(x)(isnumeric(x)||islogical(x)));
ip.addParameter('initRank', min(m,n), @isscalar); %%%%!!!ATTENTION
ip.addParameter('maxiters', 100, @isscalar);
%ip.addParameter('tol', 1e-5, @isscalar);
ip.addParameter('Gu', eye(m), @(x)(isnumeric(x)||islogical(x)));
ip.addParameter('Gv', eye(n), @(x)(isnumeric(x)||islogical(x)));
ip.addParameter('ImgOri', zeros(m,n),@(x)(isnumeric(x)||islogical(x)));

ip.parse(varargin{:});

O = ip.Results.obs;
k = ip.Results.initRank;
maxiters = ip.Results.maxiters;
%tol = ip.Results.tol;
Lu = ip.Results.Gu;
Lv = ip.Results.Gv;
X0 = ip.Results.ImgOri;
% mask_val = ip.Results.V;
%% Initialization
% M = M.*O;
Nobs = sum(O(:));
% for noise
a0 = 1e-6;
b0 = 1e-6;
tao = 1;

% for covariance
c0 = 1e-6; % noisy case
d0 = 1e-6;
lambdas = ones(k,1);

% for factors X = abc' = UV'(svd init)
% dscale of M from BCPF_TC
dscale = sum((M(:)-sum(M(:)/Nobs)).^2)/Nobs;
dscale = sqrt(dscale)/2;
M = M./dscale;  
USigma = cell(1,k);
VSigma = cell(1,k);
[USigma{:}] = deal(eye(m));
[VSigma{:}] = deal(eye(n));
% X = M;
% X(O==0) = sum(M(:))/Nobs; % for factor initialization only
% [a,b,c] = svd(double(X), 'econ');
% U = a(:,1:k)*(b(1:k,1:k)).^(0.5);
% V = (b(1:k,1:k)).^(0.5)*c(:,1:k)';
% V = V';
% X = (sum(M(:))/Nobs)*ones(m,n);

X = rand(m,n);
X = ~O.*X + M; % for factor initialization only
[a,b,c] = svds(X,k);
U = a*(b.^(0.5));
V = (b.^(0.5))*c';
V = V';

% U = rand(m,k);
% V = rand(n,k);

old_psnr = 10;
% old_perct_recovery = get_perct(M0, dscale.*U*V', idx_unknown);
    %RMSE=full(sqrt(sum(sum((Val-dscale.*X).^2.*mask_val))/nnz(Val)));
% fprintf('init_perct_recovery = %g', old_perct_recovery); 
%M = M.*O;
%% Create figures
% tbc

%% Model learning
fprintf('\n----------Learning Begin----------\n')
for it = 1:maxiters
    %% update factor matrices U and V (column by column u1, u2..., v1, v2...)
    for i = 1:k
        EUrVr = U*V' - U(:,i)*V(:,i)';
          % USigma(:,:,i) = eye(m)/((tao*diag(O*(V(:,i).*V(:,i))) + lambdas(i)*Lu) + eye(m)*(1e-6));
        USigma{i} = inv(tao*diag(O*(V(:,i).*V(:,i)+diag(VSigma{i}))) + lambdas(i)*Lu);
        %USigma(:,:,i) = pinv(tao*Evvt(:,:,i) + lambdas(i)*Lu);
        U(:,i) = tao*USigma{i}*(O.*(M - EUrVr))*V(:,i);
    end
    for i = 1:k
        EUrVr = U*V' - U(:,i)*V(:,i)';
          % VSigma(:,:,i) = eye(n)/((tao*diag(O'*(U(:,i).*U(:,i))) + lambdas(i)*Lv) + eye(n)*(1e-6));
        VSigma{i} = inv(tao*diag( O'*( (U(:,i).*U(:,i)+diag(USigma{i}) ))) + lambdas(i)*Lv);
        %VSigma(:,:,i) = pinv(tao*Euut(:,:,i) + lambdas(i)*Lv);
        V(:,i) = tao*VSigma{i}*(O.*(M - EUrVr))'* U(:,i);
    end
%     for r = 1:k
%         for i = 1:m
%             Evvt(i,i,r) = trace(diag(O(i,:))*(VSigma(:,:,r)+V(:,r)*V(:,r)'));
%         end
%         for i = 1:n
%             Euut(i,i,r) = trace(diag(O(:,i))*(USigma(:,:,r)+U(:,r)*U(:,r)'));
%         end
%     end
    %% update latent matrix
    X = U*V';
    %% update hyperparameter lambda
    ck = (0.5*(m+n) + c0)*ones(k,1);
    dk = zeros(k,1);
    for r = 1:k
        %dk(r) = d0 + 0.5.*(U(:,r)'*Lu*U(:,r) + V(:,r)'*Lv*V(:,r));
       dk(r) = d0 + 0.5.*(U(:,r)'*Lu*U(:,r) + trace(Lu*USigma{r}) + V(:,r)'*Lv*V(:,r)+ trace(Lv*VSigma{r}));
    end
    lambdas = ck./dk;
    %% update the noise tao
    ak = a0 + 0.5* Nobs;
    err = norm((O.*(M-X)),'fro').^2;
    for r=1:k
        USigmar = diag(USigma{r});
        VSigmar = diag(VSigma{r});
        for i = 1:m
            for j = 1:n
            if O(i,j)
            err = err + U(i,r).^2.*VSigmar(j) ...
                + V(j,r).^2.*USigmar(i) ...
                + USigmar(i)*VSigmar(j);
            end
            end
        end
    end

    bk = b0 + 0.5*err;
    tao = ak/bk;
    
%     Fit = 1-sqrt(sum(err(:)))/norm(M(:)); % measurement form BCPF_TC
%     SNR = 10*log10(var(X(:))*tao);
    %% prune out unnecessary components 
    if it>=1
%         sig = svd(X,'econ');
        %                 rankbound = max(sig)/10;
        %                 rank = sum(sig>rankbound);
        %         %     else
        %         step1 = linspace(100,2,10);
        %         step2 = 2*ones(1,10);
        %         step = [step1 step2];
        %
%         rankbound = max(sig)/500;
%         rank = sum(sig>rankbound);
        %     rankbound = ischange(sig,'MaxNumChanges',1);
        %     rank = find(rankbound==1) - 1;
        
        F = {U;V};
        F = cell2mat(F);
        Power = diag(F'*F);
        Pbound = max(Power)/650000;%650000;
%         rank = sum(Power>Pbound);
%         Tol = min(maxk(Power,rank));
        indices = (Power>Pbound);
        rank = sum(indices);
        U = U(:,indices);
        V = V(:,indices);
        USigma = USigma(indices);
        VSigma = VSigma(indices);
        lambdas = lambdas(indices);
        k = rank;
    end
    %% display progress
    %     perct_recovery = get_perct(M0, dscale.*X, idx_unknown);
    psnr = PSNR(dscale.*X,X0);
%     rmse=full(sqrt(sum(sum((Val-dscale.*X).^2.*mask_val))/nnz(Val)));
%     if it>3&&(psnr - old_psnr)<1e-5
%         break;
%     end
%´òÓ¡¹Øµô
    fprintf('Iter. %d: psnr = %g, Rank = %d \n', it, psnr, k); 
    old_psnr = psnr;
end

%% prepare the results
X = U*V';
X = X*dscale;
% err = norm((O.*(M-X)),'fro').^2;
% Fit = 1-sqrt(sum(err(:)))/norm(M(:));
% SNR = 10*log10(var(X(:))*tao);
%% results
Result.X = X;
% Result.SNR = SNR;
if (psnr - old_psnr)>0
    stat.psnr = psnr;
else
    stat.psnr = old_psnr;
end
Result.EstRank = k;
stat.lambdas = lambdas;
stat.tao = tao;
stat.U = U;
stat.V = V;
stat.USigma = USigma;
stat.VSigma = VSigma;
