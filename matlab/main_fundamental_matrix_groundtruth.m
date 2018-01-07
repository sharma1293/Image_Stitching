%%
%% load images and match files for the first example
%%

I1 = imread('house1.jpg');
I2 = imread('house2.jpg');
matches = load('house_matches.txt'); 
% this is a N x 4 file where the first two numbers of each row
% are coordinates of corners in the first image and the last two
% are coordinates of corresponding corners in the second image: 
% matches(i,1:2) is a point in the first image
% matches(i,3:4) is a corresponding point in the second image

N = size(matches,1);

%%
%% display two images side-by-side with matches
%% this code is to help you visualize the matches, you don't need
%% to use it to produce the results for the assignment
%%
%imshow([I1 I2]); hold on;
%plot(matches(:,1), matches(:,2), '+r');
%plot(matches(:,3)+size(I1,2), matches(:,4), '+r');
%line([matches(:,1) matches(:,3) + size(I1,2)]', matches(:,[2 4])', 'Color', 'r');
%pause;

%%
%% display second image with epipolar lines reprojected 
%% from the first image
%%

% first, fit fundamental matrix to the matches
%F = fit_fundamental_normalised(matches); % this is a function that you should write
F = fit_fundamental(matches);
L = (F * [matches(:,1:2) ones(N,1)]')'; % transform points from 
% the first image to get epipolar lines in the second image

% find points on epipolar lines L closest to matches(:,3:4)
L = L ./ repmat(sqrt(L(:,1).^2 + L(:,2).^2), 1, 3); % rescale the line
pt_line_dist = sum(L .* [matches(:,3:4) ones(N,1)],2);
residuals = abs(pt_line_dist);
fprintf('The residual error is %i',mean(residuals));
closest_pt = matches(:,3:4) - L(:,1:2) .* repmat(pt_line_dist, 1, 2);

% find endpoints of segment on epipolar line (for display purposes)
pt1 = closest_pt - [L(:,2) -L(:,1)] * 10; % offset from the closest point is 10 pixels
pt2 = closest_pt + [L(:,2) -L(:,1)] * 10;

% display points and segments of corresponding epipolar lines
clf;
imshow(I2); hold on;
plot(matches(:,3), matches(:,4), '+r');
line([matches(:,3) closest_pt(:,1)]', [matches(:,4) closest_pt(:,2)]', 'Color', 'r');
line([pt1(:,1) pt2(:,1)]', [pt1(:,2) pt2(:,2)]', 'Color', 'g');

function F = fit_fundamental(matches)
    %Taking 8 points
    matchPoints = matches(1:8,:);
    x1 = matchPoints(:,1);
    y1 = matchPoints(:,2);
    x2 = matchPoints(:,3);
    y2 = matchPoints(:,4);
    
    %a_mat = zeros();
    a_mat = [];
    for i = 1:8
        curr_a = [x2(i)*x1(i) x2(i)*y1(i) x2(i) y2(i)*x1(i) y2(i)*y1(i) y2(i) x1(i) y1(i) 1];
        a_mat = cat(1,a_mat,curr_a);
    end
    %% The Fundamental Matrix Song
    [U D V] = svd(a_mat);
    F = reshape(V(:,9),3,3)';
    [FU,FD,FV] = svd(F);
    FD(3,3) = 0;
    F  = FU*FD*FV';
end
function F = fit_fundamental_normalised(matches)
    %Taking 8 points
    matchPoints = matches(1:8,:);
    x1 = matchPoints(:,1);
    y1 = matchPoints(:,2);
    x2 = matchPoints(:,3);
    y2 = matchPoints(:,4);
    
    %Normalise the coordinates
    [Transform1 normXY1] = normaliseCoordinated([x1 y1 ones(size(x1,1),1)]);
    [Transform2 normXY2] = normaliseCoordinated([x2 y2 ones(size(x2,1),1)]);
    %a_mat = zeros();
    a_mat = [normXY2(:,1).*normXY1(:,1) normXY2(:,1).*normXY1(:,2) normXY2(:,1) normXY2(:,2).*normXY1(:,1) normXY2(:,2).*normXY1(:,2) normXY2(:,2) normXY1(:,1) normXY1(:,2) ones(size(normXY1,1),1)];
    
    %for i = 1:8
    %    curr_a = [x2(i)*x1(i) x2(i)*y1(i) x2(i) y2(i)*x1(i) y2(i)*y1(i) y2(i) x1(i) y1(i) 1];
    %    a_mat = cat(1,a_mat,curr_a);
    %end
    %% The Fundamental Matrix Song
    [U D V] = svd(a_mat);
    F = reshape(V(:,9),3,3)';
    [FU,FD,FV] = svd(F);
    FD(3,3) = 0;
    F  = FU*FD*FV';
    F = Transform2'*F*Transform1;
    %Now transforming coordinates back to unnormalised
end
function [T norm_coord] = normaliseCoordinated(input_coord)
     %normalise the x and y coordinates by dividing them by z
     input_coord(:,1) = input_coord(:,1)./input_coord(:,3);
     input_coord(:,2) = input_coord(:,2)./input_coord(:,3);
     input_coord(:,3) = 1;
     %Calculating centroids
     centroid_x = mean(input_coord(:,1));
     centroid_y = mean(input_coord(:,2));
     %shift the center of image to origin.
     shifted_coord(:,1) = input_coord(:,1) - centroid_x;
     shifted_coord(:,2) = input_coord(:,2) - centroid_y;
     %Scale the shfted points so that mean square distance between points
     %and origin is sqrt(2)
     %dist_points_origin = sqrt(shifted_coord(:,1).^2 + shifted_coord(:,2).^2 );
     mean_dist_points_origin = mean((sqrt(shifted_coord(:,1).^2 + shifted_coord(:,2).^2 )));
     sigma = sqrt(2)/mean_dist_points_origin;
     %% Reference Peter Kovesi's explanation of normalisation of coordinates
     T = [sigma 0 -sigma*centroid_x;0 sigma -sigma*centroid_y;0 0 1];
     norm_coord = T*input_coord';
     norm_coord=  norm_coord';
     
     
     
     
end

