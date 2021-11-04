% data = {'num_checkin':[],
%         'ave_checkin':[],
%         'var_checkin':[],
%         'entropy_hour':[],
%         'entropy_day':[],
%         'entropy_month':[],
%         'entropy_category':[],
%         'main_country':[],
%         'num_poi_visit':[],
%         'ave_latitude':[],
%         'ave_longitude':[],
%         'var_latitude':[],
%         'var_longitude':[],
%         'constraint':[],
%         'effective_size':[],
%         'efficiency':[],
%         'hierarchy':[],
%         'betweenness_centrality':[],
%         'degree':[],
%        }

data1 = csvread('/Users/sds/Downloads/nonSHS2.csv'); 
data2 = csvread('/Users/sds/Downloads/SHS2.csv'); 
h1=cdfplot(data1);
hold on;
h2=cdfplot(data2);
set(h1,'color','b','LineStyle','--');
set(h2,'color','r');
axis([0,200,-inf,inf])
set(gca,'YTickLabel',{'0','10','20','30','40','50','60','70','80','90','100'});
xlabel('Degree','FontSize',10);
ylabel('Percentage(%)','FontSize',10);
legend("non-SHS","SHS",'Location','SouthEast');
title("")
