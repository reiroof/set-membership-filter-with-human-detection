function f = myfun(x,W1,W2,x1,x2,n)
% X = x*W1+(1-x)*W2;
% Y=inv(X);
% Y=0.5*(Y+Y');
% k=1-x*(1-x)*(x2-x1)'*W2*Y*W1*(x2-x1);
% q=Y*(x*W1*x1+(1-x)*W2*x2);
% f=k*det(X)*trace(det(X)*Y*(W1-W2))-n*((det(X))^2)*(2*q'*W1*x1-2*q'*W2*x2+q'*(W2-W1)*q-x1'*W1*x1+x2'*W2*x2);
f = (1-x*(1-x)*((x2-x1).'*W2*inv(x*W1+(1-x)*W2)*W1*(x2-x1))) *det(x*W1 +(1-x)*W2)^2 *trace(inv(x*W1 +(1-x)*W2)*(W1-W2)) - n*det(x*W1 +(1-x)*W2)^2 * (2*(inv(x*W1 +(1-x)*W2)*(x*W1*x1+(1-x)*W2*x2)).'*(W1*x1-W2*x2)+(inv(x*W1 +(1-x)*W2)*(x*W1*x1+(1-x)*W2*x2)).'*(W2-W1)*(inv(x*W1 +(1-x)*W2)*(x*W1*x1+(1-x)*W2*x2))-x1.'*W1*x1+x2.'*W2*x2);
