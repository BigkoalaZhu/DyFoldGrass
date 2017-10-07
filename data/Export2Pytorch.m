clear all
trainingData = load('../data/trainingData_chair.mat');
data = trainingData.data;
dataNum = length(data);

maxBoxes = 30;
maxOps = 50;
maxSyms = 10;
maxDepth = 10;
copies = 1;

boxes = zeros(12, maxBoxes*dataNum*copies);
ops = zeros(maxOps,dataNum*copies);
syms = zeros(8,maxSyms*dataNum*copies);
weights = zeros(1,dataNum*copies);

for i = 1:dataNum
    p_index = i;
    
    symboxes = data{p_index}.symshapes;
    treekids = data{p_index}.treekids;
    symparams = data{p_index}.symparams;
    b = size(symboxes,2);
    l = size(treekids,1);
    box = zeros(12, b);
    op = -ones(1,l);
    sym = zeros(8,1);
    
    stack = [treekids(l, 1), treekids(l, 2)];
    op(1) = 1;
    
    while size(s,2) ~= 0
        
    end
    
    box = repmat(box, 1, copies);
    op = repmat(op, 1, copies);
    sym = repmat(sym, 1, copies);
    boxes(:, (i-1)*maxBoxes*copies+1:i*maxBoxes*copies) = box;
    ops(:,(i-1)*copies+1:i*copies) = op;
    syms(:, (i-1)*maxSyms*copies+1:i*maxSyms*copies) = sym;
    weights(:, (i-1)*copies+1:i*copies) = b/maxBoxes;
end

save('boxes.mat','boxes');
save('ops.mat','ops');
save('syms.mat','syms');
save('weights.mat','weights');