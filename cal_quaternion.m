function quaternion = cal_quaternion(R)

x = sqrt(R(1,1)-R(2,2)-R(3,3)+1)/2;
y = sqrt(-R(1,1)+R(2,2)-R(3,3)+1)/2;
z = sqrt(-R(1,1)-R(2,2)+R(3,3)+1)/2;
w = sqrt(R(1,1)+R(2,2)+R(3,3)+1)/2;
quaternion = [x y z w];

[~,I] = max(quaternion);
% I = 1;
if I == 1 
    quaternion = [ x, (R(1,2)+R(2,1))/(4*x), (R(3,1)+R(1,3))/(4*x), (R(3,2)-R(2,3))/(4*x) ];
elseif I == 2
    quaternion = [ (R(1,2)+R(2,1))/(4*y), y, (R(2,3)+R(3,2))/(4*y), (R(1,3)-R(3,1))/(4*y) ];
elseif I == 3
    quaternion = [ (R(3,1)+R(1,3))/(4*z), (R(2,3)+R(3,2))/(4*z), z, (R(2,1)-R(1,2))/(4*z) ];
elseif I == 4
    quaternion = [ (R(3,2)-R(2,3))/(4*w), (R(1,3)-R(3,1))/(4*w), (R(2,1)-R(1,2))/(4*w), w ];
end