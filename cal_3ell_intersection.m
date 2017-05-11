function [centerfin,ell_intersection] = cal_ell3_intersection(Sigma1,Sigma2,Sigma3,x1,x2,x3,n)
opt = optimset('Display','off');
s = fsolve(@myfun,0.5,opt,Sigma1,Sigma2,x1,x2,n);
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
x12 = double(inv(X)*(lamda*Sigma1*x1+(1-lamda)*Sigma2*x2));
Sigma12 = double(1/alpha * X); % inv(Q) ’ZŒa‚Æ’·Œa‚ª‚»‚Ì‚Ü‚Ü‚í‚©‚é

opt = optimset('Display','off');
s1 = fsolve(@myfun,0.5,opt,Sigma12,Sigma3,x12,x3,n);
if (s1<0)||(s1>1)
    if det(Sigma12)<det(Sigma3)
        s1=1;
    else
        s1=0;
    end
end
lamdafin = s1;
Xfin = lamdafin*Sigma12 +(1-lamdafin)*Sigma3;
alphafin = 1-lamdafin*(1-lamdafin)*((x3-x12).'*Sigma3*inv(Xfin)*Sigma12*(x3-x12));
centerfin = double(inv(Xfin)*(lamdafin*Sigma12*x12+(1-lamdafin)*Sigma3*x3));
ell_intersection = double(1/alphafin * Xfin); % inv(Q) ’ZŒa‚Æ’·Œa‚ª‚»‚Ì‚Ü‚Ü‚í‚©‚é