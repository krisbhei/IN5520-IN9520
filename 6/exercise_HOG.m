close all 

% 1.
mnist_labels=loadmnistlabels('train-labels.idx1-ubyte');
mnist_imgs=loadmnistimages('train-images.idx3-ubyte');

% 2.
num_train = 30000;
num_valid = 30000;
data_sz = size(mnist_imgs(:,1),1);

mnist_train_labels = mnist_labels(1:num_train,1);
mnist_train_imgs = mnist_imgs(:,1:num_train);
valid_labels = mnist_labels((num_train + 1):(num_train + num_valid),1);
valid_imgs = mnist_imgs(:,(num_train + 1):(num_train + num_valid));

% 3. 
figure()
for i = 1:10 
    subplot(2,5,i)
    imshow(reshape(mnist_imgs(:,i),28,28),'InitialMagnification','fit')
end

% 4. 
first_pixel = mnist_imgs(1,:);
fprintf('Variance of the first pixel at every data: %g\n', var(double(first_pixel)))

% 5.
num_classes = 10;
class_means = zeros(data_sz,num_classes);
for ii = 1:num_classes
    class_means(:,ii) = mean(mnist_train_imgs(:,mnist_train_labels == ii-1),2); 
end


estimated_imarr_labels = zeros(1, num_valid);
for ii = 1:num_valid
    [~, est_class] = min(vecnorm(bsxfun(@minus, valid_imgs(:,ii), class_means))) ; 
    estimated_imarr_labels(ii) = est_class-1;
end

fprintf('percentage of correctly classified samples using mean: %g percent\n', (sum(estimated_imarr_labels(:) == valid_labels(:))/num_valid*100) );

sz = round(sqrt(data_sz));
figure()
for ii = 1:num_classes
    subplot(2,5,ii)
    imagesc(reshape(class_means(:,ii), sz, sz))
end

% 6.
rng(5520);
cell_sz = [2,4,8];

num_train_samples = 5;
r = randi([1 num_train],1,num_train_samples);

for ii = 1:length(cell_sz)
    
    c = cell_sz(ii);
    
    figure()
    for jj = 1:num_train_samples
        [~, hogVisualization] = extractHOGFeatures(reshape(mnist_train_imgs(:,r(jj)), sz, sz), ...
                                                   'CellSize', [c, c]);
        subplot(1,num_train_samples,jj)
        plot(hogVisualization)
        title(sprintf('cell size = %d, img nr. %d',c,jj))
    end
end

figure()
for ii = 1:num_train_samples 
    subplot(1,num_train_samples ,ii)
    imagesc(reshape(mnist_train_imgs(:,r(ii)), sz, sz))
    title(sprintf('image nr. %d',ii))
end

% 7. 
c = cell_sz(1);
[hogfeatures, hogVisualization] = extractHOGFeatures(reshape(mnist_train_imgs(:,1), sz, sz), ...
                                                     'CellSize', [c, c]);

num_hogfeatures = length(hogfeatures);
hogfeatures = zeros(num_hogfeatures, num_train);
for ii = 1:num_train
    hogfeatures(:,ii) = extractHOGFeatures(reshape(mnist_train_imgs(:,ii), sz, sz), ...
                                           'CellSize', [c, c]);
end


hogfeatures_means = zeros(num_hogfeatures,num_classes);
for ii = 1:num_classes
    hogfeatures_means(:,ii) = mean(hogfeatures(:,mnist_train_labels == ii-1),2);
end

estimated_valid_labels_hog = zeros(1, num_valid);
for ii = 1:num_valid
    hogvalid = extractHOGFeatures(reshape(valid_imgs(:,ii), sz, sz), ...
                                                     'CellSize', [c, c]);
    [~, est_class] = min(vecnorm(bsxfun(@minus,hogvalid.',hogfeatures_means))) ; 
    estimated_valid_labels_hog(ii) = est_class-1;
end

fprintf('percentage of correctly classified samples using HOG: %g percent\n', (sum(estimated_valid_labels_hog(:) == valid_labels(:))/num_valid*100) );

% 8.
imarr = load('imarr.mat');
imarr = imarr.imarr ;

figure()
for ii=1:6
    subplot(2,3,ii);
    imagesc(reshape(imarr(ii*10,:,:),28,28))
    colormap gray
    title(sprintf('image nr. %d', ii*10))
end

% 9.
imarrneg = double(imcomplement(imarr));


% 10.
figure()
for ii=1:10
    subplot(2,5,ii);
    imagesc(reshape(imarrneg(90+ii,:,:),28,28))
    colormap gray
    title(sprintf('image nr. %d', 90+ii))
end

rng(5520);

% make label matrix
labels_num = [1:9 0];
labels = repelem(labels_num,10);

imarr_flat = imarrneg(:,:);

% make a choice of which samples to use for training 
indices = randperm(100);

% take num_training out, leave 100-num_training percent
num_training = 50;
num_test = 100 - num_training;

training_inds = indices(1:num_training);
test_inds = indices((num_training+1):100);

train_imarr = imarr_flat(training_inds,:); 
train_imarr_labels = labels(training_inds);

test_imarr = imarr_flat(test_inds,:);
test_imarr_labels = labels(test_inds);

figure()
for ii=1:20
    subplot(4,5,ii)
    imagesc(reshape(train_imarr(ii,:),28,28))
    title(sprintf('label nr. %d',train_imarr_labels(ii)))
end

% classify using means of the image pixels
data_sz = 28*28;

num_classes = 10;
class_means = zeros(data_sz,num_classes);
for ii = 1:num_classes
    class_means(:,ii) = mean(train_imarr(train_imarr_labels == labels_num(ii),:),1); 
end

figure()
for ii=1:num_classes
    subplot(2,5,ii)
    imagesc(reshape(class_means(:,ii),28,28))
    title(sprintf('%d',labels_num(ii)))
end

estimated_imarr_labels = zeros(1, num_test);
for ii = 1:num_test
    [~, est_class] = min(vecnorm(bsxfun(@minus, test_imarr(ii,:).', class_means))) ; 
    estimated_imarr_labels(ii) = labels_num(est_class);
end

fprintf('percentage of correctly classified imarr samples using mean: %g percent\n', (sum(estimated_imarr_labels(:) == test_imarr_labels(:))/num_test*100) );

% classify using hog
sz = 28;

c = 4;
[hogfeatures, hogVisualization] = extractHOGFeatures(reshape(train_imarr(1,:), sz, sz), ...
                                                     'CellSize', [c, c]);

num_hogfeatures = length(hogfeatures);
hogfeatures = zeros(num_hogfeatures, num_training);
for ii = 1:num_training
    hogfeatures(:,ii) = extractHOGFeatures(reshape(train_imarr(ii,:), sz, sz), ...
                                           'CellSize', [c, c]);
end


hogfeatures_means = zeros(num_hogfeatures,num_classes);
for ii = 1:num_classes
    hogfeatures_means(:,ii) = mean(hogfeatures(:,train_imarr_labels == labels_num(ii)),2);
end

estimated_imarr_labels_hog = zeros(1, num_test);
for ii = 1:num_test
    hogvalid = extractHOGFeatures(reshape(test_imarr(ii,:), sz, sz), ...
                                                     'CellSize', [c, c]);
    [~, est_class] = min(vecnorm(bsxfun(@minus,hogvalid.',hogfeatures_means))) ; 
    estimated_imarr_labels_hog(ii) = labels_num(est_class);
end

fprintf('percentage of correctly classified imarr samples using HOG: %g percent\n', (sum(estimated_imarr_labels_hog(:) == test_imarr_labels(:))/num_test*100) );
