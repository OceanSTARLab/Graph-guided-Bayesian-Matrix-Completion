function[Result, stat] = BMCG_GH(M, varargin)

%% Set parameters
[m,n] = size(M);

ip = inputParser;
ip.addParameter('init', 'ml', @(x) (ismember(x,{'ml','rand'})));
ip.addParameter('obs', ones(m,n),@(x)(isnumeric(x)||islogical(x)));
ip.addParameter('initRank', min(m,n), @isscalar); 
ip.addParameter('maxiters', 100, @isscalar);
ip.addParameter('Gu', eye(m), @(x)(isnumeric(x)||islogical(x)));
ip.addParameter('Gv', eye(n), @(x)(isnumeric(x)||islogical(x)));
ip.addParameter('ImgOri', zeros(m,n),@(x)(isnumeric(x)||islogical(x)));

ip.addParameter('dimRed', 1, @isscalar);

ip.parse(varargin{:});

init  = ip.Results.init;
O = ip.Results.obs;
k = ip.Results.initRank;
maxiters = ip.Results.maxiters;
%tol = ip.Results.tol;
Lu = ip.Results.Gu;
Lv = ip.Results.Gv;
X0 = ip.Results.ImgOri;
DIMRED   = ip.Results.dimRed;

%% Initialization
Nobs = sum(O(:));

% dscale of M from BCPF_TC
dscale = sum((M(:)-sum(M(:)/Nobs)).^2)/Nobs;
dscale = sqrt(dscale)/2;
M = M./dscale;  

%for noise
epsilon = 1e-6;
tao = 1;

% for factors X = abc' = UV'(svd init)
USigma = cell(1,k);
VSigma = cell(1,k);
[USigma{:}] = deal(eye(m));%creat a covariance matrix for every column of U
[VSigma{:}] = deal(eye(n));

X = rand(m,n);
X = ~O.*X + M; % for factor initialization only
[a,b,c] = svd(X,'econ');

switch init
    case 'ml'    % Maximum likelihood
        % [a,b,c] = svds(X,k);%it can't be done for k>max(m,n)
        if k <= max(m,n)
            U = a(:,1:k)*(b(1:k,1:k).^(0.5));
            V = (b(1:k,1:k).^(0.5))*c(:,1:k)';%
            V = V';
        else %
            U = [a*(b.^(0.5)) randn(size(a,1),k-size(a,2))];
            V = [(b.^(0.5))*c' ; randn(k-size(b,1),size(b,2))];
            V = V';
        end
    case 'rand'   % Random initialization
        U = rand(m,k);
        V = rand(n,k);
end

%for GH prior
% a0=1;
b0=0;
lambda0=8;
kappa_a1=1-lambda0/2;
kappa_a2=1e-6;
z_lambda0 = lambda0*ones(k,1);
% z_a0 = a0*ones(k,1);%
z_b0 = b0;
a_mackay = ones(k,1);

% zi~zk
gammas = ones(k,1);

inv_z_g = gammas;

old_psnr = 10;

%% Model learning
fprintf('\n----------Learning Begin----------\n')
for it=1:maxiters
    %% update factor matrices U and V (column by column u1, u2..., v1, v2...)
%     Aw = diag(gammas);%
    for i = 1:k
        EUrVr = U*V' - U(:,i)*V(:,i)';
        USigma{i} = inv(tao*diag(O*(V(:,i).*V(:,i)+diag(VSigma{i}))) + gammas(i)*Lu);
        U(:,i) = tao*USigma{i}*(O.*(M - EUrVr))*V(:,i);
    end
    for i = 1:k
        EUrVr = U*V' - U(:,i)*V(:,i)';
        VSigma{i} = inv(tao*diag( O'*( (U(:,i).*U(:,i)+diag(USigma{i}) ))) + gammas(i)*Lv);
        V(:,i) = tao*VSigma{i}*(O.*(M - EUrVr))'* U(:,i);
    end

    %% update latent matrix
    epsilon_stop = 1e-10; % this value can be set to trade-off the convergence speed and learning accuracies 
    diff=U*V' - X;
    if norm(diff(:),'fro')<=  epsilon_stop * (m*n)  %
        disp('\\\======= Converged===========\\\');
        break;
    end
    X = U*V';
    
    %% update hyperparameter z
    z_lambda = z_lambda0 - (m+n)/2*ones(k,1);
    z_b = zeros(k,1);
    for r = 1:k
        %dk(r) = d0 + 0.5.*(U(:,r)'*Lu*U(:,r) + V(:,r)'*Lv*V(:,r));
       z_b(r) = z_b0 + (U(:,r)'*Lu*U(:,r) + trace(Lu*USigma{r}) + V(:,r)'*Lv*V(:,r)+ trace(Lv*VSigma{r}));
%         z_b(r) = z_b0;
    end
    z_a = a_mackay;
    z_g = zeros(k,1);
    for i = 1 : k
        if abs(z_b(i))>= 1e-6      
            %
            z_g(i) = (sqrt(z_a(i))./sqrt(z_b(i)+eps)).*besselk(z_lambda(i)-1,sqrt(z_a(i)).*sqrt(z_b(i))+eps)./besselk(z_lambda(i),sqrt(z_a(i)).*sqrt(z_b(i))+eps);
            inv_z_g(i) = (sqrt(z_b(i))./sqrt(z_a(i))).*besselk(z_lambda(i)+1,sqrt(z_a(i)).*sqrt(z_b(i))+eps)./besselk(z_lambda(i),sqrt(z_a(i)).*sqrt(z_b(i))+eps);
        end
        if abs(z_b(i))<= 1e-6 || isnan(z_g(i))
            z_g(i) =  (sqrt(z_a(i)))./(sqrt(z_b(i))+eps);
            inv_z_g(i) = (sqrt(z_b(i) + eps)./sqrt(z_a(i)));
        end
    end    
    gammas = z_g;%
    
    for i = 1 : k
        a_mackay(i) = (kappa_a1+z_lambda0(i)/2)./(kappa_a2 + inv_z_g(i)/2);
    end
    
    %% update the noise tao
    ak = epsilon + 0.5* Nobs;
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

    bk = epsilon + 0.5*err;
    tao = ak/bk;

     %% prune out unnecessary components 
     %
    if DIMRED==1  && it >=2
        F = {U;V};
        F = cell2mat(F);
        Power = diag(F'*F);
%         Tol = (m+n)*eps(norm(F,'fro'));
        Tol = (m+n)*eps(norm(F,'fro'));
        rankest = sum(Power> Tol );
         if max(rankest)==0
            disp('Rank becomes 0 !!!');
            break;
         end  
         
        if k~= max(rankest)
            indices = Power > Tol;
            U = U(:,indices);
            V = V(:,indices);
            USigma = USigma(indices);
            VSigma = VSigma(indices);
            gammas = gammas(indices);
            k = rankest;

            z_lambda0 = z_lambda0(indices,1);
            a_mackay = a_mackay(indices, 1);
        end
        
    end

    %% display progress
    %     perct_recovery = get_perct(M0, dscale.*X, idx_unknown);
    psnr2 = PSNR(dscale.*X,X0);
%打印关掉
    fprintf('Iter. %d: psnr = %g, Rank = %d \n', it, psnr2, k); 
    old_psnr = psnr2;    

    
    
end

%% prepare the results
X = U*V';
X = X*dscale;

%% results
Result.X = X;
% Result.SNR = SNR;
if (psnr2 - old_psnr)>0
    stat.psnr = psnr2;
else
    stat.psnr = old_psnr;
end
Result.EstRank = k;
stat.tao = tao;
stat.U = U;
stat.V = V;
stat.USigma = USigma;
stat.VSigma = VSigma;
stat.lambda = gammas;