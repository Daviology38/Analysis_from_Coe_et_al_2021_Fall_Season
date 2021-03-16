clear all; close all;
tmp = load('C:\Users\CoeFamily\OneDrive - University of Massachusetts Lowell - UMass Lowell\Autumn WTs\era5_son_season_new_oct2020_cluster95\CI_results.mat');
K = tmp.K;
for i = 1:3640
   val = K(i,7);
   if(val == 1) 
       K(i,7) = 5;
   elseif(val == 2)
       K(i,7) = 1;
   elseif(val == 3)
       K(i,7) = 6;
   elseif(val == 4)
       K(i,7) = 7;
   elseif(val == 5)
       K(i,7) = 3;
   elseif(val == 6)
       K(i,7) = 4;
   else
       K(i,7) = 2;
   end
end
K = K(:,7);
xx = length(K);
k = 7;
storageyear = zeros(91,7);
storageyear3 = zeros(91,7);
y = 1;
year = 1;
%Put the clusters into an array
for i = 1:xx
    X(i) = K(i);    
end
clusteroct = zeros(1240,1);
%Take each date and put it into one array for occurrence rates
y = 1;
p = 1;
for i = 1:2911
    cluster = X(i);
   storageyear(y,cluster) = storageyear(y,cluster) + 1;
   y = y + 1;

   if( y == 92)
       y = 1;
   end
    
    
end

%Take each date and put it into one array for occurrence rates
y = 1;
for i = 1:3640
   cluster = X(i);
   storageyear3(y,cluster) = storageyear3(y,cluster) + 1;
   if( y > 30 && y <=61)
       clusteroct(p,1) = cluster;
       p = p + 1;
   end
   y = y + 1;
   if( y == 92)
       y = 1;
   end
    
    
end

save('oct_clust.mat','clusteroct')

months = {'Sep','Oct','Nov','Dec'};
storageyear2 = (storageyear ./32) .* 100;
storageyear4 = (storageyear3 ./40) .* 100;
%Put together the early and late season WTs into their own arrays
earlyseason = storageyear2(:,1) + storageyear2(:,6);
lateseason = storageyear2(:,3) + storageyear2(:,4) + storageyear2(:,7);
earlyseason2 = storageyear4(:,1) + storageyear4(:,6);
lateseason2 = storageyear4(:,3) + storageyear4(:,4) + storageyear4(:,7);
t = storageyear4(:,5);
e = storageyear4(:,2);

%Plot figure without smoothing
figure
plot(earlyseason)
hold on
plot(lateseason)
hold on
xticks([0 32 62 92])
xticklabels(months)
xlabel('Date')
ylabel('% of Days')
legend({'Early Season','Late Season'},'Location','northeast')
title('WT Occurrence Rates');
print(gcf,'-dpng',sprintf('dailyoccurrencerate.png'));

%Apply a smoother to the data, start with 5 day running mean
running = zeros(91,7);
i = 3;

while i <=89
   x = i - 2;
   xx = i - 1;
   y = i + 1;
   yy = i + 2;
   if(x == -1 || x == 0)
       x = 91 + x;
   end
   if(yy == 93 || yy == 92)
       yy = yy - 91;
   end
   if( xx == 0 )
       xx = 91;
   end
   if( y == 92)
       y = 1;
   end
   for j = 1:7
   d1 = storageyear(x,j);
   d2 = storageyear(xx,j);
   d3 = storageyear(i,j);
   d4 = storageyear(yy,j);
   d5 = storageyear(y,j);
   tot = (d1 + d2 + d3 + d4 + d5) ./ 5;
   running(i,j) = (tot./32) .* 100;
   end
   i = i + 1;
end

   
%Put together the early and late season WTs into their own arrays
earlyseasonpost = running(3:89,1) + running(3:89,6) + running(3:89,3);
lateseasonpost = running(3:89,3) + running(3:89,4) + running(3:89,7) + running(3:89,5);
transition = running(3:89,5);
extra = running(3:89,2);

running = zeros(91,7);
i = 3;

while i <=89
   x = i - 2;
   xx = i - 1;
   y = i + 1;
   yy = i + 2;
   if(x == -1 || x == 0)
       x = 91 + x;
   end
   if(yy == 93 || yy == 92)
       yy = yy - 91;
   end
   if( xx == 0 )
       xx = 91;
   end
   if( y == 92)
       y = 1;
   end
   for j = 1:7
   d1 = storageyear3(x,j);
   d2 = storageyear3(xx,j);
   d3 = storageyear3(i,j);
   d4 = storageyear3(yy,j);
   d5 = storageyear3(y,j);
   tot = (d1 + d2 + d3 + d4 + d5) ./ 5;
   running(i,j) = (tot./40) .* 100;
   end
   i = i + 1;
end

   
%Put together the early and late season WTs into their own arrays
earlyseasonpost2 = running(3:89,1) + running(3:89,6) + running(3:89,2);
lateseasonpost2 = running(3:89,3) + running(3:89,4) + running(3:89,7) + running(3:89,5);

transition2 = running(3:89,5);
extra2 = running(3:89,2);



%Plot figure with smoothing
figure
plot(earlyseasonpost2)
hold on
plot(lateseasonpost2)
xticks([0 29 59 89])
xticklabels(months)
ylim([0 100])
xlabel('Date')
ylabel('% of Days')
legend({'Early Season', 'Late Season'},'Location','northeast')
title('5-Day Running Mean WT Occurrence Rates');
print(gcf,'-dpng',sprintf('5daymeanoccurrencerate_post1999.png'));