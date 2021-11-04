data1 = csvread('/Users/sds/Downloads/csv/pagerank_new1.csv'); 
data2 = csvread('/Users/sds/Downloads/csv/pagerank_new2.csv'); 
h1=cdfplot(data1);
hold on;
h2=cdfplot(data2);
set(h1,'color','b');
set(h2,'color','r');
axis([0,0.00005,-inf,inf])
set(gca,'YTickLabel',{'0','10','20','30','40','50','60','70','80','90','100'});
xlabel('PageRank','FontSize',10);
ylabel('Percentage(%)','FontSize',10);
legend("old graph","new graph",'Location','SouthEast');
title("")