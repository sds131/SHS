%('US', 134), ('ID', 124), ('BR', 119), ('JP', 89), ('TR', 88), ('MX', 62), ('MY', 52), ('TH', 45), ('CL', 38), ('RU', 19)
%('CL', 327), ('JP', 137), ('BR', 94), ('US', 93), ('MX', 68), ('CR', 63), ('MY', 53), ('TR', 24), ('CO', 20), ('ES', 19)
%CL JP BR US MX CR MY TR
% data1=[38,89,119,134,62,0,52,88];
% data2=[327,137,94,93,68,63,53,24];
% bar(data1);
% hold on;
% bar(data2);
x=[38,327;89,137;119,94;134,93;62,68;0,63;52,53;88,24];
bar(x);
set(gca,'XTickLabel',{'CL','JP','BR','US','MX','CR','MY','TR'});
xlabel('Main Country','FontSize',10);
ylabel('number of people','FontSize',10);
legend('non-SHS','SHS');
title('')