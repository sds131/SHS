% A = [104969 9268 87
%      104321 9464 539
%      101332 10085 2907
%      89703 12435 12186
%      72358 17870 24096
%      17201 49263 47860
%      240 46566 67518];
A=[109916 4378 30
   109661 4429 234
   107982 4645 1697
   101254 5329 7741
   92445 6439 15440
   70671 12239 31414
   40894 25430 48000];
A=reshape(A,[7 1 3]);
A=A/114324;
groupLabels = {'0.01%','0.1%','1%','5%','10%','20%','30%'};     % set labels
plotBarStackGroups(A, groupLabels); % plot groups of stacked bars
axis([0,8,0,1])
set(gca,'YTickLabel',{'0','10','20','30','40','50','60','70','80','90','100'});
xlabel('Fraction of Network Removed','FontSize',10);
ylabel('Percentage(%)','FontSize',10);
legend("LCC","Middle region","Singletons");
title('')