
img1                     = imread('house1.jpg');
img2                     = imread('house2.jpg');
%matches = load('house_matches.txt'); 
%img                     = imread(imgPath);
img1Double                 = im2double(img1);
img2Double                 = im2double(img2);
img1Gray               = rgb2gray(img1Double);
img2Gray               = rgb2gray(img2Double);
numPoints = 30;
[cim_1, r_1, c_1] = harris(img1Gray, 0.5, 0.01, 2, 0);
[cim_2, r_2, c_2] = harris(img2Gray, 0.5, 0.01, 2, 0);
neighbourHood_img1 = findNeighbourhood(img1Gray,r_1,c_1,1,21);
neighbourHood_img2 = findNeighbourhood(img2Gray,r_2,c_2,1,21);
result = dist2(neighbourHood_img1,neighbourHood_img2);
[~,sortedArrayIndex] = sort(result(:),'ascend');
selectedInd = sortedArrayIndex(1:numPoints);
%[~,sortedArrayIndex] = sort(result,'ascend');
%for 
%threshold = findNthSmallestElement(result,100);
%selectedInd = find(result<=threshold);
[selectedR selectedC] = ind2sub(size(result),selectedInd);
selectedRImage1 = r_1(selectedR);
selectedCImage1 = c_1(selectedR);
selectedRImage2 = r_2(selectedC);
selectedCImage2 = c_2(selectedC);
fundamental_matrix_details = performRansacFundamentalMatrix(selectedCImage1,selectedRImage1,selectedCImage2,selectedRImage2,30,20000,30);
%fundamental_matrix_details = performRansacFundamentalMatrix(matches(:,1),matches(:,2),matches(:,3),matches(:,4),30,20000,30);
%fundamental_matrix_details = performRansacFundamentalMatrix(selectedRImage1,selectedCImage1,selectedRImage2,selectedCImage2,40,20000,30);
F = fundamental_matrix_details{1,1};
best_inliers_ind = fundamental_matrix_details{1,4};
matches = [selectedCImage1(best_inliers_ind) selectedRImage1(best_inliers_ind) selectedCImage2(best_inliers_ind) selectedRImage2(best_inliers_ind)];
N = size(matches,1);
L = (F * [matches(:,1:2) ones(N,1)]')'; % transform points from 
% the first image to get epipolar lines in the second image

% find points on epipolar lines L closest to matches(:,3:4)
L = L ./ repmat(sqrt(L(:,1).^2 + L(:,2).^2), 1, 3); % rescale the line
pt_line_dist = sum(L .* [matches(:,3:4) ones(N,1)],2);
closest_pt = matches(:,3:4) - L(:,1:2) .* repmat(pt_line_dist, 1, 2);

% find endpoints of segment on epipolar line (for display purposes)
pt1 = closest_pt - [L(:,2) -L(:,1)] * 10; % offset from the closest point is 10 pixels
pt2 = closest_pt + [L(:,2) -L(:,1)] * 10;

% display points and segments of corresponding epipolar lines
clf;
imshow(img2); hold on;
plot(matches(:,3), matches(:,4), '+r');
line([matches(:,3) closest_pt(:,1)]', [matches(:,4) closest_pt(:,2)]', 'Color', 'r');
line([pt1(:,1) pt2(:,1)]', [pt1(:,2) pt2(:,2)]', 'Color', 'g');

function selectedNeighbourhood = findNeighbourhood(imgMatrix, selected_r, selected_c,isPadding,maskSize)
    centralPixelMask = ceil(maskSize/2);
    maskSideWidth = centralPixelMask-1;

    if(isPadding)
        paddingSize = maskSideWidth;
        imgMatrix = padarray(imgMatrix,[paddingSize paddingSize],'replicate');
        selected_r = selected_r+paddingSize;
        selected_c = selected_c+paddingSize;
    end
    selectedNeighbourhood = [];
    for i = 1: size(selected_r,1)
        %currentNeighbourHood = getNeighbourhood(imgMatrix,selected_r(i),selected_c(i));
        currentNeighbourHood = imgMatrix(((selected_r(i)-maskSideWidth):(selected_r(i)+maskSideWidth)),((selected_c(i)-maskSideWidth):(selected_c(i)+maskSideWidth)));
        currentNeighbourHood = reshape(currentNeighbourHood, [1, maskSize*maskSize]);
        selectedNeighbourhood = cat(1,selectedNeighbourhood,currentNeighbourHood);
    end  
    %selectedNeighbourhood = zscore(selectedNeighbourhood')';
end
function fundamental_matrix_details = performRansacFundamentalMatrix(x1,y1,x2,y2,inliersPercent,numOfEpochs,threshold)
numPoints = size(x1,1);
numOfPointsReq = 8;
maxInliers = 0;
best_f_mat = [];%zeros(3,3);
best_residual_error = Inf;
for i = 1:numOfEpochs
    %take random 4 points
    n = randperm(numPoints,numOfPointsReq);
    select_x1 = x1(n);
    select_y1 = y1(n);
    select_x2 = x2(n);
    select_y2 = y2(n);
    curr_f = fit_fundamental_normalised([select_x1 select_y1 select_x2 select_y2]);
    %Find inliers
    xyz1 = [x1 y1 ones(numPoints,1)];
    xyz2 = [x2 y2 ones(numPoints,1)];
    L1 = (curr_f * xyz1')';
    L1 = L1 ./ repmat(sqrt(L1(:,1).^2 + L1(:,2).^2), 1, 3); % rescale the line
    %dist_error_1 = abs(dot(xyz2',L1')');
    dist_error_1 = abs(sum(L1.*xyz2,2));
    %pt_line_dist1 = sum(L1 .* [x2 y2 ones(N,1)],2);
    %closest_pt1 = [x2 y2] - L1(:,1:2) .* repmat(pt_line_dist1, 1, 2);
    %dist_error_1 = dist2(closest_pt1,[x2 y2]);
    curr_residual_error = Inf;
    %for j = 1:size(closest_pt1,1)
        
    %end
    
    L2 = (curr_f' * xyz2')';
    L2 = L2 ./ repmat(sqrt(L2(:,1).^2 + L2(:,2).^2), 1, 3); % rescale the line
    %dist_error_2 = abs(dot(L2',xyz1')');
    dist_error_2 = abs(sum(L2.*xyz1,2));
    inlier_points_ind = find(dist_error_1<threshold & dist_error_2<threshold);
    curr_num_inliers = size(inlier_points_ind,1);
    %curr_mean_residual_error = mean(dist_error_1(inlier_points_ind).^2 + dist_error_2(inlier_points_ind).^2);
    curr_residual_error = sum( dist_error_1(inlier_points_ind).^2 + dist_error_2(inlier_points_ind).^2 ) / curr_num_inliers;
    if(curr_num_inliers >= maxInliers & curr_residual_error <= best_residual_error)
        maxInliers = curr_num_inliers;
        best_residual_error = curr_residual_error;
        best_inliers_ind = inlier_points_ind;
        %best_f_mat = fit_fundamental_normalised([x1(best_inliers_ind) y1(best_inliers_ind) x2(best_inliers_ind) y2(best_inliers_ind)]);
        best_f_mat = curr_f;
    end
    %pt_line_dist2 = sum(L2 .* [x1 y1 ones(N,1)],2);
    %closest_pt2 = [x1 y1] - L2(:,1:2) .* repmat(pt_line_dist2, 1, 2);
    %dist_error_2 = dist2(closest_pt2,[x1 y1]);
    
end
if(((maxInliers/numPoints)*100)>inliersPercent)
    fprintf('Best fundamental matrix calculated for the given images is found with number of inliers:%i\n ',maxInliers);
else
    fprintf('Best fundamental matrix calculated for the given images is found with number of inliers:%i. The inlierPercentage criterai not met for best h mat.\n ',maxInliers);
end
fprintf('Best residual error is:%i\n',best_residual_error);
fundamental_matrix_details = cell(1,4);
fundamental_matrix_details{1,1} = best_f_mat;
fundamental_matrix_details{1,2} = maxInliers;
fundamental_matrix_details{1,3} = best_residual_error;
fundamental_matrix_details{1,4} = best_inliers_ind;
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
