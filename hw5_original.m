args        = argv();
trainIfile  = args{1};
testIfile   = args{2};
trainOfile  = args{3};
testOfile   = args{4};
metrics     = args{5};
num_epoch   = str2num(args{6});
HiddenUnit  = str2num(args{7});
initFlag    = str2num(args{8});
lr          = str2num(args{9});

% number of classes
ClassNumber = 10;
%% readng CVS files
function [x,y] = readCVS(trainIfile)
a = csvread(trainIfile,0,0);
y = a(:,1);
x = a(:,2:end);
end

%% -- Backward
function [g_alpha, g_beta] = NNBackward(x,y,beta,o)
% bringing the variables from object in scope
y_hat   = o.y_hat;
a       = o.a;
z       = o.z;
b       = o.b;

% calculating the gradients
gb      = y_hat - y;
g_beta  = gb*z'; % z_0 = 1
gz      = beta'*gb;
ga      = gz.*z.*(1-z);
ga(1)   = []; % remove the ga(1) which is for x_0 = 1
g_alpha = ga*x';

end

%% Forward
function o = NNForward(alpha,beta,x,y)
a     = alpha*x;
z     = 1./(1+exp(-a));
z     = [1;z];
b     = beta*z;
denom = sum(exp(b));
y_hat = exp(b)./denom;
loss = -y'*log(y_hat);
o.a = a;
o.z = z;
o.b = b;
o.y_hat = y_hat;
o.loss = loss;

end

% Error calc
function [error] = calcError(predicted,true)
error = 0;
[N,~] = size(predicted);
for i = 1:N
    if predicted(i) ~= true(i)
        error = error +1;
    end
end
error = error/N;
end

% prediction
function [predicted] = predict(features,labels,alpha,beta,ClassNumber)
predicted = [];
[N,~]= size(labels);
for n=1:N
    x  = features(n,:);
    x  = [1 x]';
    indexLabel  =labels(n)+1;
    y  = zeros(ClassNumber,1);
    y(indexLabel) = 1;
    o = NNForward(alpha,beta,x,y);
    [~,index] = max(o.y_hat);
    predicted =[predicted;index-1];
end
end


%% -- reading the input file
[features,labels] = readCVS(trainIfile);
[N,M] = size(features);
if initFlag == 1
    %alpha = random('Uniform',-0.1,0.1,HiddenUnit,M+1);
    %beta  = random('Uniform',-0.1,0.1,ClassNumber,HiddenUnit+1);
     alpha = unifrnd(-0.1,0.1,HiddenUnit,M+1);
     beta  = unifrnd(-0.1,0.1,ClassNumber,HiddenUnit+1);
else
    alpha = zeros(HiddenUnit,M+1);
    beta  = zeros(ClassNumber,HiddenUnit+1);
end
%fprintf("N = %f \n",N)
mt = fopen(metrics,'w');
%% --- Training data
for epoch=1:num_epoch
    for n=1:N
        x  = features(n,:);
        x  = [1 x]';
        indexLabel  =labels(n)+1;
        y  = zeros(ClassNumber,1);
        y(indexLabel) = 1;
        o = NNForward(alpha,beta,x,y);
        %fprintf('%3.5f\n',o.loss)
        %loss =loss+o.loss;
        [g_alpha, g_beta] = NNBackward(x,y,beta,o);
        beta  = beta  -lr.*g_beta;
        alpha = alpha -lr.*g_alpha;
    end
    loss = 0.0;
    % Training Entropy
    for n=1:N
        x  = features(n,:);
        x  = [1 x]';
        indexLabel  =labels(n)+1;
        y  = zeros(ClassNumber,1);
        y(indexLabel) = 1;
        o = NNForward(alpha,beta,x,y);
        %fprintf('%3.5f\n',o.loss)
        loss =loss+o.loss;
    end
    %tr = fopen(metrics, 'a');
    fprintf(mt,"epoch=%d crossentropy(train): %3.4f\n",epoch, loss/N);
    % Test data Entropy
    [features_T,labels_T]=readCVS(testIfile);
    [N_t,~] = size(features_T);
    loss = 0.0;
    for n=1:N_t
        x  = features_T(n,:);
        x  = [1 x]';
        indexLabel  =labels_T(n)+1;
        y  = zeros(ClassNumber,1);
        y(indexLabel) = 1;
        o = NNForward(alpha,beta,x,y);
        %fprintf('%3.5f\n',o.loss)
        loss =loss+o.loss;
    end
    fprintf(mt,"epoch=%d crossentropy(test): %3.4f\n",epoch, loss/N_t);
end
fclose(mt);



%% ------ Prediction Time--------

[predicted_tr] = predict(features,labels,alpha,beta,ClassNumber);
[trError] = calcError(predicted_tr,labels);

ft = fopen(trainOfile,'a');
[N,~] = size(predicted_tr);
for i=1:N
    fprintf(ft,"%d\n",predicted_tr(i));
end
fclose(ft);


[features_T,labels_T]=readCVS(testIfile);
[predicted_tst] = predict(features_T,labels_T,alpha,beta,ClassNumber);
[tstError] = calcError(predicted_tst,labels_T);

ftest = fopen(testOfile,'a');
[N_t,~] = size(predicted_tst);
for i=1:N_t
    fprintf(ftest,"%d",predicted_tst(i));
    fprintf(ftest,"\n");
end
fclose(ftest);


%% --writing the metrics
mt = fopen(metrics,'a');
fprintf(mt,"error(train): %2.4f\n",trError);
fprintf(mt,"error(test): %2.4f\n",tstError);
fclose(mt);

