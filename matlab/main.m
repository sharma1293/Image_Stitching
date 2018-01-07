
img1                     = imread('..\data\part1\uttower\left.jpg');
img2                     = imread('..\data\part1\uttower\right.jpg'); 

img1Double                 = im2double(img1);
img2Double                 = im2double(img2);
img1Gray               = rgb2gray(img1Double);
img2Gray               = rgb2gray(img2Double);
numPoints = 200;
[cim_1, r_1, c_1] = harris(img1Gray, 2, 0.05, 2, 0);
[cim_2, r_2, c_2] = harris(img2Gray, 2, 0.05, 2, 0);
neighbourHood_img1 = findNeighbourhood(img1Gray,r_1,c_1,1,21);
neighbourHood_img2 = findNeighbourhood(img2Gray,r_2,c_2,1,21);
result = dist2(neighbourHood_img1,neighbourHood_img2);
[~,sortedArrayIndex] = sort(result(:),'ascend');
selectedInd = sortedArrayIndex(1:numPoints);

[selectedR selectedC] = ind2sub(size(result),selectedInd);
selectedRImage1 = r_1(selectedR);
selectedCImage1 = c_1(selectedR);
selectedRImage2 = r_2(selectedC);
selectedCImage2 = c_2(selectedC);
h_mat_detail = performRansac(selectedRImage1,selectedCImage1,selectedRImage2,selectedCImage2,50,40000,30);
h_mat = h_mat_detail{1,1};
maxInliers = h_mat_detail{1,2};
best_inlier_ind = h_mat_detail{1,3};
best_residual_error = h_mat_detail{1,4};
fprintf('Best residual error is:%i\n ',best_residual_error);
figure; imshow([img1Double img2Double]); hold on; title('Inliers');
hold on; plot(selectedCImage1(best_inlier_ind), selectedRImage1(best_inlier_ind),'ro'); plot(selectedCImage2(best_inlier_ind) + size(img1Double,2), selectedRImage2(best_inlier_ind), 'ro'); 

t = maketform('projective',h_mat');
imgFinal = stitch_images(img1Double,img2Double,t);
figure,imshow(imgFinal);

function overlaped_img = stitch_images(img1,img2,T)
[y1 x1 z1] = size(img1);
[y2 x2 z2] = size(img2);
[img1Transformed xdata1 ydata1] =imtransform(img1,T);
xdata_f=[min(1,xdata1(1)) max(x2,xdata1(2))];
ydata_f=[min(1,ydata1(1)) max(y2,ydata1(2))];
%Now transform the images to the fit in overlapped image
img_f_1=imtransform(img1,T,'XData',xdata_f,'YData',ydata_f);
img_f_2=imtransform(img2,maketform('affine',eye(3)),'XData',xdata_f,'YData',ydata_f);

overlapInd = img_f_1 & img_f_2;
img_f_1(overlapInd) = img_f_1(overlapInd)./2;
img_f_2(overlapInd) = img_f_2(overlapInd)./2;
overlaped_img = img_f_1+img_f_2;
end
%{
function img_final = stich_images(img1,img2,T)
[y1 x1 z1] = size(img1);
[y2 x2 z2] = size(img2);
[img1Transformed xdata1 ydata1] =imtransform(img1,T);
xdataf = [min(1,xdata1(1)) max(x2,xdata1(2))];
ydataf = [min(1,ydata1(1)) max(y2,ydata1(2))];
img_final = [];
if(z2>1 && z2>1)
    for i= 1:z1
        current_overlapped_img = overlapImages(xdataf,ydataf,img1(:,:,i),img1Transformed(:,:,i),img2(:,:,i));
        img_final = cat(3,img_final,current_overlapped_img);
    end
else
    img_final = overlapImages(xdataf,ydataf,img1,img1Transformed,img2);
end
end

 function overlaped_img = overlapImages(xdataf,ydataf,img1,img1Transformed,img2)
    [y1 x1 z1] = size(img1);
    [y2 x2 z2] = size(img2);
    x_final_img = round(abs(xdataf(2))+abs(xdataf(1)));
    y_final_img = round(abs(ydataf(2))+abs(ydataf(1)));
    img_f_1 = zeros(y_final_img,x_final_img);
    img_f_2 = zeros(y_final_img,x_final_img);
    overlaped_img = zeros(y_final_img,x_final_img);
    img_f_2(y_final_img-y2+1:end,x_final_img-x2+1:end) = img2;
    img_f_1(1:size(img1Transformed,1),1:size(img1Transformed,2)) = img1Transformed;
    overlapInd = img_f_1 & img_f_2;
    img_f_1(overlapInd) = img_f_1(overlapInd)./2;
    img_f_2(overlapInd) = img_f_2(overlapInd)./2;
    overlaped_img = img_f_1+img_f_2;
 end
%}
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
    %Scaling each feature vector to zero mean and unit variance. 
    selectedNeighbourhood = selectedNeighbourhood';
    selectedNeighbourhood = (selectedNeighbourhood - repmat(mean(selectedNeighbourhood), size(selectedNeighbourhood,1), 1)) ./ repmat(std(selectedNeighbourhood), size(selectedNeighbourhood,1), 1);
    selectedNeighbourhood = selectedNeighbourhood';
end

function h_mat_details = performRansac(y1,x1,y2,x2,inliersPercent,numOfEpochs,threshold)
%Initialize variables
numPoints = size(x1,1);
numOfPointsReq = 4;
maxInliers = 0;
best_h_mat = [];%zeros(2*numOfPointsReq,9);
for i = 1:numOfEpochs
    %take random 4 points
    n = randperm(numPoints,numOfPointsReq);
    select_x1 = x1(n);
    select_y1 = y1(n);
    select_x2 = x2(n);
    select_y2 = y2(n);
    %h = calculateHomographyMatrix(select_x1,select_y1,select_x2,select_y2);
    h = calculateHomographyMatrix(select_x1,select_y1,select_x2,select_y2);
    h = reshape(h,3,3)';
    %Calculate number of inliers
    diffn = setdiff([1:numPoints],n);
    inlier_details = calcInliers([x1(diffn) y1(diffn) ones(size(diffn,2),1)], [x2(diffn) y2(diffn) ones(size(diffn,2),1)],h,threshold);
    numInliers = inlier_details{1,1};
    curr_inlier_ind = inlier_details{1,2};
    curr_residual_error = inlier_details{1,3};
    
    if(numInliers>maxInliers)
        maxInliers = numInliers;
        best_h_mat = h;
        best_inlier_ind = curr_inlier_ind;
        best_residual_error = curr_residual_error;
    end    
end
h_mat_details = cell(1,4);
h_mat_details{1,1} = best_h_mat;
h_mat_details{1,2} = maxInliers;
h_mat_details{1,3} = best_inlier_ind;
h_mat_details{1,4} = best_residual_error;
if(((maxInliers/numPoints)*100)>inliersPercent)
    fprintf('Best homography matrix calculated for the given images is found with number of inliers:%i\n ',maxInliers);
else
    fprintf('Best homography matrix calculated for the given images is found with number of inliers:%i. The inlierPercentage criterai not met for best h mat.\n ',maxInliers);
end
end

function h_mat = calculateHomographyMatrix(x1,y1,x2,y2)%(x2,y2,x1,y1)%
numOfPointsReq = size(x1,1);
a_mat = zeros(2*numOfPointsReq,9);
for i = 1:numOfPointsReq
    %curr_a_mat = [-x1(i) -y1(i) -1 0 0 0 x2(i)*x1(i) x2(i)*y1(i) x2(i);0 0 0 -x1(i) -y1(i) -1 x1(i)*y2(i) -y1(i)*y2(i) -y2(i)];
    curr_a_mat = [x1(i) y1(i) 1 0 0 0 -x1(i).*x2(i) -y1(i).*x2(i) -x2(i);0 0 0 x1(i) y1(i) 1 -x1(i).*y2(i) -y1(i).*y2(i) -y2(i)];
    %curr_a_mat = [0 0 0 x1(i) y1(i) 1 -x1(i).*y2(i) -y1(i).*y2(i) -y2(i);x1(i) y1(i) 1 0 0 0 -x1(i).*x2(i) -y1(i).*x2(i) -x2(i)];
    %a_mat = cat(1,a_mat,curr_a_mat)
    k = i*2;
    a_mat(k-1:k,:) = curr_a_mat;
end
%a_mat = [-x1 -y1 -1 0 0 0 x2*x1 x2*y1 x2;0 0 0 -x1 -y1 -1 x1*y2 -y1*y2 -y3]
a_mat;
[U S V] = svd(a_mat);
size(V);
h_mat = V(:,end);%./V(end,end);
end
%Calculate the projected coordinates of image 1 from homographic matrix,
%and then subtract them from image 2 coordinates. Square the result to
%calculate the error. Now if the error is less than threshold, then count
%them as inliers.
function inliers_details = calcInliers(img1coord,img2coord,h,threshold)
proj_coord_img1 = h*img1coord';%img1coord'.*h;
proj_coord_img1 = proj_coord_img1';
proj_coord_img1(:,1) = proj_coord_img1(:,1)./proj_coord_img1(:,3);
proj_coord_img1(:,2) = proj_coord_img1(:,2)./proj_coord_img1(:,3);
proj_coord_img1(:,3) = proj_coord_img1(:,3)./proj_coord_img1(:,3);

x_err = proj_coord_img1(:,1) - img2coord(:,1);
y_err = proj_coord_img1(:,1) - img2coord(:,1);
error_mat = (x_err.^2)+(y_err.^2);
inliers_ind =find(error_mat<threshold);
numInliers = size(inliers_ind,1);
inliers_details = cell(1,3);
inliers_details{1,1} = numInliers;
inliers_details{1,2} = inliers_ind;
inliers_residual_error = sum(error_mat(inliers_ind));
inliers_details{1,3} = inliers_residual_error;
end
