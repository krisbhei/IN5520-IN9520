
%Read MNIST data and labels
mnist1trainandvalidlab=loadmnistlabels('train-labels.idx1-ubyte');
mnist1trainandvalidim=loadmnistimages('train-images.idx3-ubyte');

%Divide training set into training and validation part

mnist1trainlab = mnist1trainandvalidlab(1:30000,1);
mnist1trainim = mnist1trainandvalidim(:,1:30000);
validlab = mnist1trainandvalidlab(30001:60000,1);
validim = mnist1trainandvalidim(:,30001:60000);

mnisttestim = loadmnistlabels('t10k-images.idx3-ubyte');
mnisttestlab = loadmnistimages('t10k-labels.idx1-ubyte');

nofclasses = 10;

[nbands, ntrainsamp] = size(mnist1trainim);
[nbands, ntestsamp] = size(mnisttestim);
[nbands, nvalidsamp] =size(validim);
sizex = sqrt(nbands);
sizey = sizex;
%size(mnist1trainim)
mnistim = mnist1trainim';
mnistvalidim = validim';
mnisttestim = mnisttestim';

%Organize data for training
klass0lab = mnist1trainlab(mnist1trainlab==0);

klass1lab = mnist1trainlab(mnist1trainlab==1);
klass2lab = mnist1trainlab(mnist1trainlab==2);
klass3lab = mnist1trainlab(mnist1trainlab==3);
klass4lab = mnist1trainlab(mnist1trainlab==4);
klass5lab = mnist1trainlab(mnist1trainlab==5);
klass6lab = mnist1trainlab(mnist1trainlab==6);
klass7lab = mnist1trainlab(mnist1trainlab==7);
klass8lab = mnist1trainlab(mnist1trainlab==8);
klass9lab = mnist1trainlab(mnist1trainlab==9);

klass0data = mnistim(mnist1trainlab==0,:);
klass1data = mninstim(mnist1trainlab==1,:);
klass2data = mninstim(mnist1trainlab==2,:);
klass3data = mninstim(mnist1trainlab==3,:);
klass4data = mninstim(mnist1trainlab==4,:);
klass5data = mninstim(mnist1trainlab==5,:);
klass6data = mninstim(mnist1trainlab==6,:);
klass7data = mninstim(mnist1trainlab==7,:);
klass8data = mninstim(mnist1trainlab==8,:);
klass9data = mninstim(mnist1trainlab==9,:);





% % %If running KNN on original features, use this code:
% % 
% % trainX = [klass0data; klass1data; klass2data; klass3data; klass4data; klass5data; klass6data; klass7data; klass8data; klass9data ];
% % trainY = [klass0lab; klass1lab; klass2lab; klass3lab; klass4lab; klass5lab; klass6lab; klass7lab; klass8lab; klass9lab]
% % size(trainX);
% % size(trainY);
% % tabulate(trainY);
% % 
% % prior = [0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1];
% % classNames = {'0','1','2','3','4','5','6','7','8','9'}; % Class order
% % %Mdl = fitcnb(trainX,trainY,'ClassNames',classNames,'Prior',prior)
% % 
% % Mdl = fitcknn(trainX,trainY,'NumNeighbors',3,'Standardize',0)
% % %Mdl.prior = prior;
% % 
% % 
% % 
% % 
% % validX = validim;
% % validY = validlab;
% % 
% % %knnlabel = predict(Mdl,testX);
% % %save('label_knn5.mat','label');
% % %cmat = confusionmat(testY,label);
% % 
% % %knnaccuracy = sum(knnlabel==testY)/length(testY);








sizex = sqrt(nbands);
sizey = sizex;

mean0vec = zeros(1,nbands);
mean1vec = zeros(1,nbands);
mean2vec = zeros(1,nbands);
mean3vec = zeros(1,nbands);
mean4vec = zeros(1,nbands);
mean5vec = zeros(1,nbands);
mean6vec = zeros(1,nbands);
mean7vec = zeros(1,nbands);
mean8vec = zeros(1,nbands);
mean9vec = zeros(1,nbands);




for b=1:nbands
        
        mean0vec(b) = mean(klass0data(:,b));
        mean1vec(b) = mean(klass1data(:,b));
        mean2vec(b) = mean(klass2data(:,b));
        mean3vec(b) = mean(klass3data(:,b));
        mean4vec(b) = mean(klass4data(:,b));
        mean5vec(b) = mean(klass5data(:,b));
        mean6vec(b) = mean(klass6data(:,b));
        mean7vec(b) = mean(klass7data(:,b));
        mean8vec(b) = mean(klass8data(:,b));
        mean9vec(b) = mean(klass9data(:,b));
end
mean0 = reshape(mean0vec, 28,28); 
mean1 = reshape(mean1vec, 28,28); 
mean2 = reshape(mean2vec, 28,28); 
mean3 = reshape(mean3vec, 28,28); 
mean4 = reshape(mean4vec, 28,28); 
mean5 = reshape(mean5vec, 28,28); 
mean6 = reshape(mean6vec, 28,28); 
mean7 = reshape(mean7vec, 28,28); 
mean8 = reshape(mean8vec, 28,28); 
mean9 = reshape(mean9vec, 28,28); 

meanvecs = zeros(10,nbands);
meanvecs(1,:) = mean0vec;
meanvecs(2,:) = mean1vec;
meanvecs(3,:) = mean2vec;
meanvecs(4,:) = mean3vec;
meanvecs(5,:) = mean4vec;
meanvecs(6,:) = mean5vec;
meanvecs(7,:) = mean6vec;
meanvecs(8,:) = mean7vec;
meanvecs(9,:) = mean8vec;
meanvecs(10,:) = mean9vec;






%A classifier using variance will fail because many of the pixels are 0 in
%all images, so the variance will be zero




% Select one image per class and study HOG-histograms visually for a grid
% size. Try different grid sizes. 


cellsize = 8;
currim = reshape(mnist1trainim(:,1), 28,28);
[featureVector,hogVisualization] = extractHOGFeatures(currim,'CellSize',[cellsize cellsize]);
featuresize = length(featureVector);
feattrainvec = zeros(ntrainsamp,featuresize);
%vissize = size(hogVisualization);
%vistrainvec = zeros(ntrainsamp, vissize);
for m=1:ntrainsamp
      currim = reshape(mnist1trainim(:,m),28,28);
      [currfeat,currvis] = extractHOGFeatures(currim, 'CellSize', [cellsize cellsize]);
      feattrainvec(m,:) = currfeat;
      %To display HOG features:
      %figure(1)
      %imshow(currim)
      %figure(2)
      %plot(currvis)
end



%Now retrain using HOG features
      
featklass0 = feattrainvec(mnist1trainlab==0,:);
featklass1 = feattrainvec(mnist1trainlab==1,:);
featklass2 = feattrainvec(mnist1trainlab==2,:);
featklass3 = feattrainvec(mnist1trainlab==3,:);
featklass4 = feattrainvec(mnist1trainlab==4,:);
featklass5 = feattrainvec(mnist1trainlab==5,:);
featklass6 = feattrainvec(mnist1trainlab==6,:);
featklass7 = feattrainvec(mnist1trainlab==7,:);
featklass8 = feattrainvec(mnist1trainlab==8,:);
featklass9 = feattrainvec(mnist1trainlab==9,:);

featmeanvec0= zeros(1,featuresize);
featmeanvec1= zeros(1,featuresize);
featmeanvec2= zeros(1,featuresize);
featmeanvec3= zeros(1,featuresize);
featmeanvec4= zeros(1,featuresize);
featmeanvec5= zeros(1,featuresize);
featmeanvec6= zeros(1,featuresize);
featmeanvec7= zeros(1,featuresize);
featmeanvec8= zeros(1,featuresize);
featmeanvec9= zeros(1,featuresize);

for b=1:featuresize
    featmeanvec0(b) = mean(featklass0(:,b));
    featmeanvec1(b) = mean(featklass1(:,b));
    featmeanvec2(b) = mean(featklass2(:,b));
    featmeanvec3(b) = mean(featklass3(:,b));
    featmeanvec4(b) = mean(featklass4(:,b));
    featmeanvec5(b) = mean(featklass5(:,b));
    featmeanvec6(b) = mean(featklass6(:,b));
    featmeanvec7(b) = mean(featklass7(:,b));
    featmeanvec8(b) = mean(featklass8(:,b));
    featmeanvec9(b) = mean(featklass9(:,b));
    
end
featmeans = zeros(10,featuresize);
featmeans(1,:) = featmeanvec0;
featmeans(2,:) = featmeanvec1;
featmeans(3,:) = featmeanvec2;
featmeans(4,:) = featmeanvec3;
featmeans(5,:) = featmeanvec4;
featmeans(6,:) = featmeanvec5;
featmeans(7,:) = featmeanvec6;
featmeans(8,:) = featmeanvec7;
featmeans(9,:) = featmeanvec8;
featmeans(10,:) = featmeanvec9;



%The loop below classifies both the original features and the HOG features
%using minimum Euclidean distance


featklasslab = zeros(nvalidsamp,1);
origklasslab = zeros(nvalidsamp,1);
featvalidvec = zeros(nvalidsamp,featuresize);
nclasses = 10;
featdists = zeros(1,nclasses);
for s=1:nvalidsamp
    currvec = mnistvalidim(s,:);
    currim = reshape(currvec, 28,28);
    [currfeat, currvis] = extractHOGFeatures(currim, 'CellSize', [cellsize, cellsize]);
    for cl=1:nclasses
        featdist(1,cl) = sum((currfeat-featmeans(cl,:)).^2);
    end
    featvalidvec(s,:) = currfeat;
    dist0 = sum(sum(reshape(currvec,28,28)-mean0).^2);
    dist1 = sum(sum(reshape(currvec,28,28)-mean1).^2);
    dist2 = sum(sum(reshape(currvec,28,28)-mean2).^2);
    dist3 = sum(sum(reshape(currvec,28,28)-mean3).^2);
    dist4 = sum(sum(reshape(currvec,28,28)-mean4).^2);
    dist5 = sum(sum(reshape(currvec,28,28)-mean5).^2);
    dist6 = sum(sum(reshape(currvec,28,28)-mean6).^2);
    dist7 = sum(sum(reshape(currvec,28,28)-mean7).^2);
    dist8 = sum(sum(reshape(currvec,28,28)-mean8).^2);
    dist9 = sum(sum(reshape(currvec,28,28)-mean9).^2);
    
    distarr = [dist0; dist1; dist2; dist3; dist4; dist5; dist6; dist7; dist8; dist9];
    
    [bestval,bestclass] = min(distarr);
    bestclass = bestclass-1;
    
    [bestfeatval,bestfeatclass] = min(featdist);
   
    bestfeatclass = bestfeatclass-1;
    origklasslab(s,1) = bestclass;
    featklasslab(s,1) = bestfeatclass;
    
    %bestclass
    %bestfeatclass
    %mnisttestlab(s,1)
    %figure(1);
    %imshow(currim);
    %figure(2);
    %plot(currvis);
    %tt=1;
    
end

    


%Classify validation data 
klassifiedlab = zeros(nvalidsamp,1);
for s= 1:nvalidsamp
    
    
    
    currvec = mnistvalidim(s,:);
    currim = reshape(currvec, 28, 28);
    
    dist0 = sum(sum(reshape(currvec,28,28)-mean0).^2);
    dist1 = sum(sum(reshape(currvec,28,28)-mean1).^2);
    dist2 = sum(sum(reshape(currvec,28,28)-mean2).^2);
    dist3 = sum(sum(reshape(currvec,28,28)-mean3).^2);
    dist4 = sum(sum(reshape(currvec,28,28)-mean4).^2);
    dist5 = sum(sum(reshape(currvec,28,28)-mean5).^2);
    dist6 = sum(sum(reshape(currvec,28,28)-mean6).^2);
    dist7 = sum(sum(reshape(currvec,28,28)-mean7).^2);
    dist8 = sum(sum(reshape(currvec,28,28)-mean8).^2);
    dist9 = sum(sum(reshape(currvec,28,28)-mean9).^2);
    
    distarr = [dist0; dist1; dist2; dist3; dist4; dist5; dist6; dist7; dist8; dist9];
    
    [bestval,bestclass] = min(distarr);
    bestclass = bestclass-1;
    
    

    

    klassifiedlab(s,1) = bestclass;
    
   
    tt=1;
end


mindistaccuracy = sum(klassifiedlab==validlab)/length(validlab);


featmindistaccuracy= sum(featklasslab==validlab)/length(validlab);

mindistaccuracy
featmindistaccuracy


%%KNN classification of the HOG-features
 %featMdl = fitcknn(feattrainvec,mnist1trainlab,'NumNeighbors',3,'Standardize',0); 
 %featknnlab = predict(featMdl, featvalidvec);
 %featknnaccuracy = sum(featknnlab==validlab)/length(validlab);

% Read handwritten images
mytest = load('imarr.mat');
nytestim = mytest.imarr;

[ns, xx,yy  ] = size(nytestim);
nylabels = zeros(100,1);
nylabels(1:10,1) = 1;
nylabels(11:20,1) = 2;
nylabels(21:30,1) = 3;
nylabels(31:40,1) = 4;
nylabels(41:50,1) = 5;
nylabels(51:60,1) = 6;
nylabels(61:70,1) = 7;
nylabels(71:80,1) = 8;
nylabels(81:90,1) = 9;
nylabels(91:100,1) = 0;
correctknn = 0;
correctfeatknn = 0;
correctmindist = 0;
correctfeatmindist = 0;
for s= 1:ns
    currim = imcomplement(squeeze(nytestim(s,:,:)));
    
    
    dist0 = sum(sum(currim-mean0).^2);
    dist1 = sum(sum(currim-mean1).^2);
    dist2 = sum(sum(currim-mean2).^2);
    dist3 = sum(sum(currim-mean3).^2);
    dist4 = sum(sum(currim-mean4).^2);
    dist5 = sum(sum(currim-mean5).^2);
    dist6 = sum(sum(currim-mean6).^2);
    dist7 = sum(sum(currim-mean7).^2);
    dist8 = sum(sum(currim-mean8).^2);
    dist9 = sum(sum(currim-mean9).^2);
    
    distarr = [dist0; dist1; dist2; dist3; dist4; dist5;dist6; dist7; dist8;dist9];
    [obestval,obestclass] = min(distarr);
    
    [currfeat,currvis] = extractHOGFeatures(currim, 'CellSize', [cellsize cellsize]);
    for cl=1:nclasses
        featdist(1,cl) = sum((currfeat-featmeans(cl,:)).^2);
    end
    [fbestval,fbestclass] = min(featdist);
    %featknncurr = predict(featMdl, currfeat);
    %origknncurr = predict(origMdl, reshape(currim,1, 784))
    
    % Correcting for indexes1-10 vs classes 0-9
    fbestclass = fbestclass-1;
    obestclass = obestclass-1;
    if fbestclass==nylabels(s,1)
        correctfeatmindist = correctfeatmindist + 1;
    end
    if obestclass == nylabels(s,1)
        correctmindist = correctmindist + 1;
    end
    %if featknncurr==nylabels(s,1)
    %    correctfeatknn = correctfeatknn + 1;
    %end
    %if origknncurr ==nylabels(s,1)
    %    correctknn = correctknn +1 ;
    %end
    
    
   
   tt=1;
end

correctmindist/ns
%correctknn/ns
correctfeatmindist/n
%correctfeatknn/ns

