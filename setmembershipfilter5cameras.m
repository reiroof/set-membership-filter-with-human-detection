clear all
close all
clc
% Before running this simulation, 
% please make the folder whose name is 'result1', 'result2', ..., 'resultN'
% in the current folder.

%%%%%%%%%%%%%%%%%%%%%%%% Settings of parameter %%%%%%%%%%%%%%%%%%%%%%%%%%%%
n = 5;   % number of cameras
gam = 0;   % location of environment
tmax = 121;   % maximal number of counter
nix = 320;   % pixel number: X-direction  
niy = 240;   % pixel number: Y-direction
lam(1:n,1) = 3.4/1000*ones(n,1);   % focal length of cameras [m]
xmax = 3.6*10^(-3)/2;  % image plane size: X-direction [m]
ymax = 2.7*10^(-3)/2;  % image plane size: Y-direction [m]
pixel = xmax/(nix/2);  % size of one pixel [m]
% vertexes of an image plane
vtx = [[xmax;ymax;lam] [-xmax;ymax;lam] [-xmax;-ymax;lam] [xmax;-ymax;lam]];
e12 = [1 0 0;0 1 0];
e1=[1 0 0];
e2=[0 1 0];
e3=[0 0 1];
xn=6;
yn=2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%% camera position and pose setting %%%%%%%%%%%%%%%%%%%%%%%%%
% camera's coordinate
% pwi(:,1) = [-4.97668; -4.50021; 5.4266]; % z 2.7 [m]
pwi(:,1) = [-7; -2; 1.5]; % z 2.7 [m]
pwi(:,2) = [0; -9; 1.6];
% pwi(:,3) = [5.5;-5.2;1.7];%[4.5; 3; 1.5];
% pwi(:,4) = [4.5; 3; 1.5];%[-4.7;4;1.5];
pwi(:,3) = [5.8;-5;1.5];
pwi(:,4) = [5;1;1.6];%[4.5; 3; 1.5];%[4;-5.5;1.5];
pwi(:,5) = [-3.4;2.4;1.4];
% pwi(:,4) = [0;0;23];
% pwi(:,1) = [-3.02114; -4.18998;2.92885];
% pwi(:,2) = [-2; -4.5; 2.7];
% pwi(:,3) = [2; 2.5; 2.7];
% pwi(:,4) = [-2; 2.5; 2.7];

% Initial condition of camera's pose (rotation matrix)
Rx1 = makehgtform('xrotate',80/180 * pi);
% Rx1 = makehgtform('xrotate',45/180 * pi);
Ry1 = makehgtform('yrotate',0/180 * pi);
Rz1 = makehgtform('zrotate',-90/180 * pi);

Rx2 = makehgtform('xrotate',80/180 * pi);
% Rx1 = makehgtform('xrotate',45/180 * pi);
Ry2 = makehgtform('yrotate',0/180 * pi);
Rz2 = makehgtform('zrotate',0/180 * pi);

% Rx3 = makehgtform('xrotate',80/180 * pi);
% Rz3 = makehgtform('zrotate',60/180 * pi);
% 
% Rx4 = makehgtform('xrotate',80/180 * pi);
% Rz4 = makehgtform('zrotate',130/180 * pi);

Rx3 = makehgtform('xrotate',80/180 * pi);
Rz3 = makehgtform('zrotate',60/180 * pi);


Rx4 = makehgtform('xrotate',80/180 * pi);
Rz4 = makehgtform('zrotate',120/180 * pi);


Rx5 = makehgtform('xrotate',80/180 * pi);
Rz5 = makehgtform('zrotate',-140/180 * pi);

% Rx3 = makehgtform('xrotate',80/180 * pi);
% Rz3 = makehgtform('zrotate',130/180 * pi);
% 
% Rx4 = makehgtform('xrotate',80/180 * pi);
% Rz4 = makehgtform('zrotate',-140/180 * pi);
% 
% 
% Rx5 = makehgtform('xrotate',80/180 * pi);
% Rz5 = makehgtform('zrotate',60/180 * pi);

G1 = Rz1*Ry1*Rx1;
R1 = G1(1:3,1:3); %単位なし
Rwi{1} = zeros(3,3*(tmax+1));   
Rwi{1}(1:3,1:3) = R1;

G2 = Rz2*Ry2*Rx2;
R2 = G2(1:3,1:3);
Rwi{2} = zeros(3,3*(tmax+1));   
Rwi{2}(1:3,1:3) = R2;

G3 = Rz3*Ry1*Rx3;
R3 = G3(1:3,1:3);
Rwi{3} = zeros(3,3*(tmax+1));   
Rwi{3}(1:3,1:3) = R3;

G4 = Rz4*Ry1*Rx4;
R4 = G4(1:3,1:3);
Rwi{4} = zeros(3,3*(tmax+1));   
Rwi{4}(1:3,1:3) = R4;

G5 = Rz5*Ry1*Rx5;
R5 = G5(1:3,1:3);
Rwi{5} = zeros(3,3*(tmax+1));   
Rwi{5}(1:3,1:3) = R5;

% Initial condition of camera's pose (quaternion)
qu = zeros(n,4);
for i = 1:n
    qu(i,:) = cal_quaternion(Rwi{i}(1:3,1:3));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ell_scale=zeros(6,6,n);
ell_posi=zeros(6,n);
ell_rot=zeros(6,6,n);
ell_qu=zeros(n,4);
for_blender = zeros(2,4,n);
%初期値を設定
for i = 1:n
    ell_scale(:,:,i) = [10 0 0 0 0 0;
        0 10 0 0 0 0;
        0 0 10 0 0 0;
        0 0 0 1.5 0 0; 
        0 0 0 0 1.5 0;
        0 0 0 0 0 1]; %[m^2]
    ell_posi(:,i) = [0;
        -1.5;
        0;
        0;
        0;
        0]; %[m]
    ell_rot(:,:,i) = eye(6); %単位なし
    ell_qu(i,:) = cal_quaternion(ell_rot(:,:,i));
    % Initial data which is sent to Blender
    step_mat = [1, zeros(1,3)];
    for_blender_camera = [step_mat; qu];
    for_blender(:,:,i) = [step_mat;ell_qu(1,:)];
    % "for_blender" is sent to Blender and this matrix is composed like
    % following: [current step number, current camera's ID (k), zeros(1,2); quaternion of camera's pose]

%     dlmwrite('data\ell_rot.txt',ell_rot,'delimiter','\t','newline','pc')
%     dlmwrite('data\ell_scale.txt',ell_scale,'delimiter','\t','newline','pc')
    fileposi = ['data\ell_posi_',num2str(i),num2str(i+1),'.txt'];
    filequ = ['data\ell_qu_',num2str(i),num2str(i+1),'.txt'];
    filescale = ['data\ell_scale_out_',num2str(i),num2str(i+1),'.txt'];
    dlmwrite(fileposi,ell_posi(:,i),'delimiter','\t','newline','pc')
    dlmwrite(filequ,for_blender(:,:,i),'delimiter','\t','newline','pc')
    dlmwrite(filescale,2*sqrt(ell_scale(:,:,i)),'delimiter','\t','newline','pc')
end

dlmwrite('data\camera.txt',for_blender_camera,'delimiter','\t','newline','pc')
%%%%%% Making the text file in which the environment's data is stored %%%%%
param = [n, nix, niy, xmax*2*10^3, ymax*2*10^3];
camera_posi_lam = [pwi' lam*10^3];


dlmwrite('data\param.txt',param,'delimiter','\t','newline','pc')
dlmwrite('data\posi_lam.txt',camera_posi_lam,'delimiter','\t','newline','pc')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%% Settings of image processing (human detection) %%%%%%%%%%%%%%%%%%%%%%%
% peopleDetector = vision.PeopleDetector('MinSize',[128 64],'MaxSize',[160 116],...
%     'ClassificationModel','UprightPeople_128x64','ScaleFactor',1.05);
peopleDetector = vision.PeopleDetector('ClassificationModel','UprightPeople_96x48','ClassificationThreshold',1,'ScaleFactor',1.05,'MinSize',[],'MaxSize',[]);
% shapeInserter = vision.ShapeInserter('BorderColor','Custom','CustomBorderColor',255);[205 97 155]
shapeInserter = vision.ShapeInserter('BorderColor','Custom','CustomBorderColor',[225 0 178],'LineWidth',2);
scoreInserter = vision.TextInserter('Text',' %f','Color', 125,'LocationSource','Input port','FontSize',16);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Waiting until the Blender's program runs
% When you complete to run the Blender's program, please press any key  
% in the MATLAB command window
pause on 
pause
figure;
R1 = R1.';
R2 = R2.';
R3 = R3.';
R4 = R4.';
R5 = R5.';
Ra(:,:,1) = R1;
Ra(:,:,2) = R2;
Ra(:,:,3) = R3;
Ra(:,:,4) = R4;
Ra(:,:,5) = R5;
count = 0;
x_hat_k1=zeros(6,n);
Sigma=zeros(6,6,n);
P = zeros(6,89,5);
Sigma_k_k=zeros(6,6,n);
x_hat_k=zeros(6,n);
y=zeros(2,n);
ya=zeros(2,n);
Sigma_k1_k1=zeros(6,6,n);
Sigma_k1_k=zeros(6,6,n);
R_hat=zeros(2,2,n);
C=zeros(2,6,n);
alphap = zeros(2,1);
alpham = zeros(2,1);
g=zeros(2,1);
timenotupdate=zeros(89,5);
timeupdate=zeros(89,5);
rt=300;


% loop of time step

for t = 1:rt
    count = count + 1
%     tic
    for k = 1:n  % loop of camera
        not_read = true;
        R = Ra(:,:,k); % temporaly variable for rotation matrix
        p = pwi(:,k); % temporaly variable for position vector
%         tic
%         reading image file which is created by Blender
        filename = ['Camera',num2str(k),'\image',num2str(count),'.png'];
%         filename = ['image\image',num2str(count),'.png'];
%         saving image file which is created by MATLAB 
%         (in that image, quadrilateral which is created by human detection image processing is drew.)
        resultname = ['result',num2str(k),'\image',num2str(count),'.png'];
        while not_read
            try
                frame = feval(@imread, filename);
                not_read = false;
            catch
                pause(0.1)
            end
        end
%         human detection is done here.
        [bboxes, scores] = step(peopleDetector, frame);
        Nim = 0;
        % bboxses is the vertexes of quadrilateral which is created by human
        % detection image processing.
        % By using these vertexes, you will estimate human 3D position.      

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%set-membership filter %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
% 
%         X=[bboxes(1)+bboxes(3)/2 bboxes(2)+bboxes(4)/2];
%         plot(X);
%         hold on;
        A = [1 0 0 1/30 0 0;
                0 1 0 0 1/30 0;
                0 0 1 0 0 1/30;
                0 0 0 1 0 0;
                0 0 0 0 1 0;
                0 0 0 0 0 1];
%         w = [0;0;0;1;1;1];
        %%%%%%%%%%%%%%%%%%% ESMF prediction step %%%%%%%%%%%%%%%%%%%%%%%
        Sigma_k_k(:,:,k) = (ell_rot(:,:,k)*ell_scale(:,:,k)*ell_rot(:,:,k).'); %[m^2]
        x_hat_k(:,k) = A*ell_posi(:,k); %[m;m/s]
        a = 0.5;
%         Q_hat = [0.0001 0 0 0 0 0;
%             0 0.0001 0 0 0 0;
%             0 0 0.0001 0 0 0;
%             0 0 0 0.000001 0 0;
%             0 0 0 0 0.0001 0;
%             0 0 0 0 0 0.0001]*50;
        Q_hat = [0.0001 0 0 0 0 0;
            0 0.0001 0 0 0 0;
            0 0 0.0001 0 0 0;
            0 0 0 0.0001 0 0;
            0 0 0 0 0.0001 0;
            0 0 0 0 0 0.0001]/100;
%         Q_hat = zeros(6,6); %[m^2]
        beta = sqrt(trace(Q_hat))/(sqrt(trace(Q_hat))+sqrt(trace(A*Sigma_k_k(:,:,k)*A.')));
        Sigma_k1_k(:,:,k) = A*Sigma_k_k(:,:,k)*A.'/(1-beta) + Q_hat/beta;%[m^2]
%         timepre(t,k)=toc
            %%%%%%%%%%%%%%% ESMF update step %%%%%%%%%%%%%%%%%%%%%%%%
        % If the k's camera detects human, enter this loop and 
        % draw quadrilaterals.
        if isempty(bboxes) == 0 
%             tic
            frame = step(shapeInserter, frame, int32(bboxes));
            b=1.7;
            y(:,k)=[bboxes(1)+bboxes(3)/2-160;-(bboxes(2)+bboxes(4)/2-120)]*pixel; %[m]
            R_hat(:,:,k) = [((bboxes(3)))^2 0;
                0 ((bboxes(4)))^2]*pixel^2; %[m^2]
            
            


            C(:,:,k) = -1/((R(3,1)*x_hat_k(1,k)+R(3,2)*x_hat_k(2,k)+R(3,3)*x_hat_k(3,k)-e3*R*p)^2)*[(R(1,1)*R(3,2)-R(3,1)*R(1,2))*x_hat_k(2,k)+(R(1,1)*R(3,3)-R(3,1)*R(1,3))*x_hat_k(3,k)-(R(1,1)*e3*R*p-R(3,1)*e1*R*p) (R(2,1)*R(3,2)-R(3,1)*R(2,2))*x_hat_k(2,k)+(R(2,1)*R(3,3)-R(3,1)*R(2,3))*x_hat_k(3,k)-(R(2,1)*e3*R*p-R(3,1)*e2*R*p);
            (R(1,2)*R(3,1)-R(3,2)*R(1,1))*x_hat_k(1,k)+(R(1,2)*R(3,3)-R(3,2)*R(1,3))*x_hat_k(3,k)-(R(1,2)*e3*R*p-R(3,2)*e1*R*p) (R(2,2)*R(3,1)-R(3,2)*R(2,1))*x_hat_k(1,k)+(R(2,2)*R(3,3)-R(3,2)*R(2,3))*x_hat_k(3,k)-(R(2,2)*e3*R*p-R(3,2)*e2*R*p);
            (R(1,3)*R(3,1)-R(3,3)*R(1,1))*x_hat_k(1,k)+(R(1,3)*R(3,2)-R(3,3)*R(1,2))*x_hat_k(2,k)-(R(1,3)*e3*R*p-R(3,3)*e1*R*p) (R(2,3)*R(3,1)-R(3,3)*R(2,1))*x_hat_k(1,k)+(R(2,3)*R(3,2)-R(3,3)*R(2,2))*x_hat_k(2,k)-(R(2,3)*e3*R*p-R(3,3)*e2*R*p);
            0 0; 
            0 0;
            0 0].'*lam(1,1); %単位なし

            ya(:,k) = y(:,k)-h(x_hat_k(:,k),lam(1,1),R,p)+C(:,:,k)*x_hat_k(:,k);
            ra = 1/2*sqrt(R_hat);
            for i = 1:2
                alphap(i) = (ya(i,k)-C(i,:,k)*x_hat_k(:,k)+ra(i,i,k))/(sqrt(C(i,:,k)*Sigma_k1_k(:,:,k)*C(i,:,k).'));
                alpham(i) = (ya(i,k)-C(i,:,k)*x_hat_k(:,k)-ra(i,i,k))/(sqrt(C(i,:,k)*Sigma_k1_k(:,:,k)*C(i,:,k).'));
                alphap(i) = min(alphap(i),1);
                alpham(i) = max(alpham(i),-1);
%                 alphap(i) * alpham(i) <= -1/xn;
                if alphap(i) * alpham(i) <= -1/xn
                    x_hat_k1(:,k) = x_hat_k(:,k);
                    Sigma_k1_k1(:,:,k) = Sigma_k1_k(:,:,k);
                    Sigma_k1_k(:,:,k) = Sigma_k1_k1(:,:,k);
                    x_hat_k(:,k)=x_hat_k1(:,k);
                else
                    g(i) = C(i,:,k)*Sigma_k1_k(:,:,k)*C(i,:,k).';
                    e(i) = sqrt(g(i))*(alphap(i)+alpham(i))/2;
                    d(i) = sqrt(g(i))*(alphap(i)-alpham(i))/2;
%                     opt = optimset('Display','off');
%                     syms x
%                     eqn = (xn-1)*g(i)^2*x^2+((2*xn-1)*d(i)^2-g(i)+e(i)^2)*g(i)*x+(xn*(d(i)^2-e(i)^2)-g(i))*d(i)^2==0;
%                     l = double(solve(eqn,x));
                    p = [(xn-1)*g(i)^2 ((2*xn-1)*d(i)^2-g(i)+e(i)^2)*g(i) (xn*(d(i)^2-e(i)^2)-g(i))*d(i)^2];
                    l = roots(p);
%                     opt = optimset('Display','off');
%                     l = fzero(@update,[0,1],opt,e(i),d(i),g(i),xn);
                    if l(1)>0
                        lambdam = l(1);
                    else 
                        lambdam = l(2);
                    end
                    lambdam;
                    S(:,:,i) = Sigma_k1_k(:,:,k) - lambdam/(d(i)^2+lambdam*g(i))*Sigma_k1_k(:,:,k)*C(i,:,k).'*C(i,:,k)*Sigma_k1_k(:,:,k); 
                    x_hat_k1(:,k) = x_hat_k(:,k)+lambdam*S(:,:,i)*C(i,:,k).'*e(i)/(d(i)^2);
                    Sigma_k1_k1(:,:,k) = (1+lambdam-lambdam*e(i)^2/(d(i)^2+lambdam*g(i)))*S(:,:,i);
                    Sigma_k1_k(:,:,k) = Sigma_k1_k1(:,:,k);
                    x_hat_k(:,k)=x_hat_k1(:,k);
                end
            end
            Sigma(:,:,k) = Sigma_k1_k1(:,:,k);
            x_hat_k1(:,k) = x_hat_k1(:,k);
%             timeupdate(t,k)=toc
        else
%             tic
            x_hat_k1(:,k) = x_hat_k(:,k);%[m]
            Sigma(:,:,k) = Sigma_k1_k(:,:,k);%[m^2]
%             timenotupdate(t,k)=toc
        end
%         tic
        imshow(frame)
        imwrite(frame, resultname)
%         timeshow(t,k)=toc
%         t3=cputime-t1
    end
    tic
    [qxfin(:,2),Sigmafin(:,:,2)]=cal_3ell_intersection(Sigma(:,:,1),Sigma(:,:,2),Sigma(:,:,3),x_hat_k1(:,1),x_hat_k1(:,2),x_hat_k1(:,3),xn);
    timeintersection(t,k)=toc
    [qxfin(:,3),Sigmafin(:,:,3)]=cal_3ell_intersection(Sigma(:,:,2),Sigma(:,:,3),Sigma(:,:,4),x_hat_k1(:,2),x_hat_k1(:,3),x_hat_k1(:,4),xn);
    [qxfin(:,4),Sigmafin(:,:,4)]=cal_3ell_intersection(Sigma(:,:,3),Sigma(:,:,4),Sigma(:,:,5),x_hat_k1(:,3),x_hat_k1(:,4),x_hat_k1(:,5),xn);
    [qxfin(:,5),Sigmafin(:,:,5)]=cal_3ell_intersection(Sigma(:,:,4),Sigma(:,:,5),Sigma(:,:,1),x_hat_k1(:,4),x_hat_k1(:,5),x_hat_k1(:,1),xn);
    [qxfin(:,1),Sigmafin(:,:,1)]=cal_3ell_intersection(Sigma(:,:,5),Sigma(:,:,1),Sigma(:,:,2),x_hat_k1(:,5),x_hat_k1(:,1),x_hat_k1(:,2),xn);
%     t2(t)=toc
    step_mat = [count+1, zeros(1,3)];
    
%     Sigmafin=Sigma;
%     qxfin=x_hat_k1;
    
    %%% ここまでで1-2 2-3 3-4 4-5の楕円ができるので，それぞれを描画する
%     tic
    Sigma_i(:,:,1)=inv(Sigmafin(:,:,1));
    lastSigma(:,:,1) = Sigma_i(1:3,1:3,1)-1/4*Sigma_i(4:6,1:3,1).'*inv(Sigma_i(4:6,4:6,1))*Sigma_i(4:6,1:3,1);
%     time_3dim(t)=toc
    [ER3_12,ES3_12] = eig(inv(lastSigma(:,:,1)));
    [ER_12,ES_12] = eig(Sigmafin(:,:,1));
    ell_posi(:,1) = qxfin(:,1);%[m]
    ell_qu(1,:) = cal_quaternion(ER3_12);
    ell_rot(:,:,1) = ER_12;%単位なし
    ell_scale(:,:,1) = ES_12; %[m^2]
    ell_scale_out(:,:,1) = 2*sqrt(ES3_12); %[m]
    for_blender(:,:,1) = [step_mat; ell_qu(1,:)];
    P(:,t,1) = ell_posi(:,1);
    dlmwrite('data\ell_qu_12.txt',for_blender(:,:,1),'delimiter','\t','newline','pc')
    dlmwrite('data\ell_posi_12.txt',ell_posi(:,1),'delimiter','\t','newline','pc')
    dlmwrite('data\ell_scale_out_12.txt',ell_scale_out(:,:,1),'delimiter','\t','newline','pc')
    
    Sigma_i(:,:,2)=inv(Sigmafin(:,:,2));
    lastSigma(:,:,2) = Sigma_i(1:3,1:3,2)-1/4*Sigma_i(4:6,1:3,2).'*inv(Sigma_i(4:6,4:6,2))*Sigma_i(4:6,1:3,2);
    [ER3_23,ES3_23] = eig(inv(lastSigma(:,:,2)));
    [ER_23,ES_23] = eig(Sigmafin(:,:,2));
    ell_posi(:,2) = qxfin(:,2);%[m]
    ell_qu(2,:) = cal_quaternion(ER3_23);
    ell_rot(:,:,2) = ER_23;%単位なし
    ell_scale(:,:,2) = ES_23; %[m^2]
    ell_scale_out(:,:,2) = 2*sqrt(ES3_23); %[m]
    for_blender(:,:,2) = [step_mat; ell_qu(2,:)];
    P(:,t,2) = ell_posi(:,2);
    dlmwrite('data\ell_qu_23.txt',for_blender(:,:,2),'delimiter','\t','newline','pc')
    dlmwrite('data\ell_posi_23.txt',ell_posi(:,2),'delimiter','\t','newline','pc')
    dlmwrite('data\ell_scale_out_23.txt',ell_scale_out(:,:,2),'delimiter','\t','newline','pc')
    
    
    Sigma_i(:,:,3)=inv(Sigmafin(:,:,3));
    lastSigma(:,:,3) = Sigma_i(1:3,1:3,3)-1/4*Sigma_i(4:6,1:3,3).'*inv(Sigma_i(4:6,4:6,3))*Sigma_i(4:6,1:3,3);
    [ER3_34,ES3_34] = eig(inv(lastSigma(:,:,3)));
    [ER_34,ES_34] = eig(Sigmafin(:,:,3));
    ell_posi(:,3) = qxfin(:,3);%[m]
    ell_qu(3,:) = cal_quaternion(ER3_34);
    ell_rot(:,:,3) = ER_34;%単位なし
    ell_scale(:,:,3) = ES_34; %[m^2]
    ell_scale_out(:,:,3) = 2*sqrt(ES3_34); %[m]
    for_blender(:,:,3) = [step_mat; ell_qu(3,:)];
    P(:,t,3) = ell_posi(:,3);
    dlmwrite('data\ell_qu_34.txt',for_blender(:,:,3),'delimiter','\t','newline','pc')
    dlmwrite('data\ell_posi_34.txt',ell_posi(:,3),'delimiter','\t','newline','pc')
    dlmwrite('data\ell_scale_out_34.txt',ell_scale_out(:,:,3),'delimiter','\t','newline','pc')
    
    
    Sigma_i(:,:,4)=inv(Sigmafin(:,:,4));
    lastSigma(:,:,4) = Sigma_i(1:3,1:3,4)-1/4*Sigma_i(4:6,1:3,4).'*inv(Sigma_i(4:6,4:6,4))*Sigma_i(4:6,1:3,4);
    [ER3_45,ES3_45] = eig(inv(lastSigma(:,:,4)));
    [ER_45,ES_45] = eig(Sigmafin(:,:,4));
    ell_posi(:,4) = qxfin(:,4);%[m]
    ell_qu(4,:) = cal_quaternion(ER3_45);
    ell_rot(:,:,4) = ER_45;%単位なし
    ell_scale(:,:,4) = ES_45; %[m^2]
    ell_scale_out(:,:,4) = 2*sqrt(ES3_45); %[m]
    for_blender(:,:,4) = [step_mat; ell_qu(4,:)];
    P(:,t,4) = ell_posi(:,4);
    dlmwrite('data\ell_qu_45.txt',for_blender(:,:,4),'delimiter','\t','newline','pc')
    dlmwrite('data\ell_posi_45.txt',ell_posi(:,4),'delimiter','\t','newline','pc')
    dlmwrite('data\ell_scale_out_45.txt',ell_scale_out(:,:,4),'delimiter','\t','newline','pc')
    
    Sigma_i(:,:,5)=inv(Sigmafin(:,:,5));
    lastSigma(:,:,5) = Sigma_i(1:3,1:3,5)-1/4*Sigma_i(4:6,1:3,5).'*inv(Sigma_i(4:6,4:6,5))*Sigma_i(4:6,1:3,5);
    [ER3_51,ES3_51] = eig(inv(lastSigma(:,:,5)));
    [ER_51,ES_51] = eig(Sigmafin(:,:,5));
    ell_posi(:,5) = qxfin(:,5);%[m]
    ell_qu(5,:) = cal_quaternion(ER3_51);
    ell_rot(:,:,5) = ER_51;%単位なし
    ell_scale(:,:,5) = ES_51; %[m^2]
    ell_scale_out(:,:,5) = 2*sqrt(ES3_51); %[m]
    for_blender(:,:,5) = [step_mat; ell_qu(5,:)];
    P(:,t,5) = ell_posi(:,5);
    dlmwrite('data\ell_qu_56.txt',for_blender(:,:,5),'delimiter','\t','newline','pc')
    dlmwrite('data\ell_posi_56.txt',ell_posi(:,5),'delimiter','\t','newline','pc')
    dlmwrite('data\ell_scale_out_56.txt',ell_scale_out(:,:,5),'delimiter','\t','newline','pc')
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
t = 1:116;
figure(1)
plot(t,P(:,:,1));
figure(2)
plot(t,P(:,:,2));
figure(3)
plot(t,P(:,:,3));
figure(4)
plot(t,P(:,:,4));
figure(5)
plot(t,P(:,:,5));
% % 
% rmdir('Camera1','s')
% rmdir('Camera2','s')
% rmdir('Camera3','s')
% rmdir('Camera4','s')
% rmdir('Camera5','s')