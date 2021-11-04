data1 = [105091,18,17,13,13,12,11,11,11,10];
data2 = [109960,17,12,11,10,9,9,8,7,7]; 
data1=log10(data1);
bar(data1);
%data2=log10(data2);
%bar(data2);
set(gca,'XTickLabel',{'1st','2nd','3rd','4th','5th','6th','7th','8th','9th','10th'});
set(gca,'YTickLabel',{'10^0','10^1','10^2','10^3','10^4','10^5','10^6'});
%set(gca,'yscale','log')
xlabel('Top 10 Connected Components','FontSize',10);
ylabel('Component Size','FontSize',10);
title('')