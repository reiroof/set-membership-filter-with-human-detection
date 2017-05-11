function [qxfin,ell_intersection] = cal_2ell_intersection(Sigma1,Sigma2,x1,x2,n)
opt = optimset('Display','off');
% x0=[0 1];
[s fval] = fzero(@myfun,0.5,opt,Sigma1,Sigma2,x1,x2,n);
if (s<0)||(s>1)
    if det(Sigma1)<det(Sigma2)
        s=1;
    else
        s=0;
    end
end
lamda = s;
X = lamda*Sigma1 +(1-lamda)*Sigma2;
alpha = 1-lamda*(1-lamda)*((x2-x1).'*Sigma2*inv(X)*Sigma1*(x2-x1));
qxfin = double(inv(X)*(lamda*Sigma1*x1+(1-lamda)*Sigma2*x2));
ell_intersection = double(1/alpha * X); % inv(Q) ’ZŒa‚Æ’·Œa‚ª‚»‚Ì‚Ü‚Ü‚í‚©‚é