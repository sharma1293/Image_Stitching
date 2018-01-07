I1 = imread('house1.jpg');
I2 = imread('house2.jpg');
matches = load('house_matches.txt'); 
P1 = load('house1_camera.txt');
P2 = load('house2_camera.txt');
c1 = calculateCenters(P1);
c2 = calculateCenters(P2);
%Calculate triangulation points
%For triangulation use equation 
%x1 cross product P1*X = 0
%so we need to find the null space of x1 cross P1
xyz1 = [matches(:,1) matches(:,2) ones(size(matches,1))];
xyz2 = [matches(:,3) matches(:,4) ones(size(matches,1))];
xyz_triangulated = [];
xyz_projected1 = [];
xyz_projected2 = [];
for i = 1:size(matches,1)
   %using equation
   x1_cross_matrix = [0 -xyz1(i,3) xyz1(i,2);xyz1(i,3) 0 -xyz1(i,1);-xyz1(i,2) xyz1(i,1) 0];
   x2_cross_matrix = [0 -xyz2(i,3) xyz2(i,2);xyz2(i,3) 0 -xyz2(i,1);-xyz2(i,2) xyz2(i,1) 0]; 
   x1_cross_p1 = x1_cross_matrix * P1;
   x2_cross_p2 = x2_cross_matrix * P2;
   a_mat = [x1_cross_p1;x2_cross_p2];
   [U S V] = svd(a_mat);
   curr_triangulated_point = V(:,end);
    
   curr_triangulated_point_cart= [];
   curr_triangulated_point_cart(1,:) = curr_triangulated_point(1,:)/curr_triangulated_point(4,:);
   curr_triangulated_point_cart(2,:) = curr_triangulated_point(2,:)/curr_triangulated_point(4,:);
   curr_triangulated_point_cart(3,:) = curr_triangulated_point(3,:)/curr_triangulated_point(4,:);
   xyz_triangulated = cat(1,xyz_triangulated,curr_triangulated_point_cart(1:3,:)');
   
   projected_points_P1 = (P1 * curr_triangulated_point);
   
   projected_points_P1(1,:) = projected_points_P1(1,:)/projected_points_P1(3,:);
   projected_points_P1(2,:) = projected_points_P1(2,:)/projected_points_P1(3,:);
   xyz_projected1 = cat(1,xyz_projected1,projected_points_P1(1:2,:)');
   
   projected_points_P2 = (P2 * curr_triangulated_point);
   
   projected_points_P2(1,:) = projected_points_P2(1,:)/projected_points_P2(3,:);
   projected_points_P2(2,:) = projected_points_P2(2,:)/projected_points_P2(3,:);
   xyz_projected2 = cat(1,xyz_projected2,projected_points_P2(1:2,:)');
   
end
%plot points
figure; axis equal;  hold on; 
plot3(-xyz_triangulated(:,1), xyz_triangulated(:,2), xyz_triangulated(:,3), '.b');
plot3(-c1(1), c1(2), c1(3),'*r');
plot3(-c2(1), c2(2), c2(3),'*g');
grid on; xlabel('x'); ylabel('y'); zlabel('z'); axis equal;

fprintf('Residuals of image 1: %i\n',mean(diag(dist2(matches(:,1:2), xyz_projected1))));
fprintf('Residuals of image 2: %i\n',mean(diag(dist2(matches(:,3:4), xyz_projected2))));

function C_cart = calculateCenters(P)
[U S V] = svd(P);
C = V(:,end);
C(1,:) = C(1,:)/C(4,:);
C(2,:) = C(2,:)/C(4,:);
C(3,:) = C(3,:)/C(4,:);
C_cart = C(1:3,:)'; 
end