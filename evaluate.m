load('prob-label.mat')
scores = squeeze(prob);
correctfactor =0;
scores(:,1)=scores(:,1)-mean(scores(:,1))-correctfactor; % 0.4 for MRI
scores(:,2)=scores(:,2)-mean(scores(:,2))+correctfactor;
[~,predlabel] = max(scores,[],2);

annos = readtable('sSCDpSCDwHC.csv');
subjects = cellfun(@(s1)strcat(s1,'.mat'),annos.Subject, 'UniformOutput', false);
labels = 1+contains(annos.Group, {'pSCD'});

ACC = sum(predlabel==labels)/76;

TP = sum(predlabel~=1 & labels~=1);
TN = sum(predlabel==1 & labels==1);
FP = sum(predlabel~=1 & labels==1);
FN = sum(predlabel==1 & labels~=1);

SEN = TP/(TP+FN);
SPE = TN/(TN+FP);
PPV = TP/(TP+FP);
F1S = 2*SEN*PPV/(SEN+PPV);
[X,Y,T,AUC] = perfcurve(labels*2-3, scores(:,2), 1);

disp([ AUC, ACC, SEN, SPE, F1S]);
   