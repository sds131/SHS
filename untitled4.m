x=1:1:20;%x轴上的数据，第一个值代表数据开始，第二个值代表间隔，第三个值代表终止
a=[4493,78733,736667,3119217,6411342,7404102,5747039,3332732,1506943,576564,209434,75903,25263,8176,3264,1402,460,52,4,0]; %a数据y值
%b=[2180,40468,348191,1163276,1950943,1962650,1255953,522309,160581,43614,11246,2806,782,367,248,131,29,0,0,0]
b=[12560,219939,1858771,7345633,14452358,15317372,9719795,3888163,1126632,294069,73201,17187,4804,2587,1877,1310,376,34,0,0];
sum(a)
sum(b)
a=a/sum(a)
b=b/sum(b)
plot(x,a,'-*b',x,b,'-or'); %线性，颜色，标记
axis([1,20,0,0.3])  %确定x轴与y轴框图大小
set(gca,'XTick',[2:2:20]) %x轴范围1-6，间隔1
set(gca,'YTick',[0:0.05:0.3]) %y轴范围0-700，间隔100
set(gca,'YTickLabel',{'0','5','10','15','20','25','30'});
xlabel('Path Length')  %x轴坐标描述
ylabel('Percentage(%)') %y轴坐标描述
legend("old graph","new graph");
