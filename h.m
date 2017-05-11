function estimate2 = h(x,lam,R1,pwi)
% x is state
% lam is  focal length of the camera
% R is rotate matrix
% pwi is location of the camera
D1 = [1 0;
    0 1;
    0 0].';
D2 = [1 0 0 0 0 0;
    0 1 0 0 0 0;
    0 0 1 0 0 0];
D3 = [0 0 1];
estimate2 = lam*(D1 * (R1*(D2 * x - pwi)))/(-(D3 * (R1 *(D2 * x - pwi))));