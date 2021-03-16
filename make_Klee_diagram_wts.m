%This script creates the Klee Diagram for our WT distribution over the
%years

%Load in the data
%data_temp = csvread('Coe_clusters_new.csv');
%clusters = data_temp(2:end,2);
tmp = load('C:\Users\CoeFamily\OneDrive - University of Massachusetts Lowell - UMass Lowell\WTs_general\era5_mam_jan21_85\CI_results.mat');
K = tmp.K;
clusters = K(:,7);

%Reshape to 91 dates per year for all years
clustersnew = reshape(clusters,92,[]);

%Plot the data
figure;hAxes = gca;
imagesc(hAxes,clustersnew);
colormap( hAxes , [1 0.9 0.1; 1 0.7 0.4; 0.4 0.7 1; 0.4 0.3 0.8; 0.4 0.6 0.6; 1 0.5 0.2; 1 0.6 0.4;] )
h = colorbar;
xticks([1 5 10 15 20 25 30 35 40])
xticklabels({'1979', '1983','1988','1993','1998','2003','2008','2013','2018'});
xtickangle(90);
xlabel('Years');
ylabel('Calendar Date');
set(h,'Ticks',[1.5 2.25 3 3.8 4.75 5.6 6.55],'TickLabels',["1" "2" "3" "4" "5" "6" "7"])
title('WTs per Year by Calendar Date');

%Now split the clusters by early and late season
%Plot the data
figure;hAxes = gca;
imagesc(hAxes,clustersnew);
colormap( hAxes , [1 1 1;  1 1 1;  1 0.4 0.3; 1 1 1; 1 1 1; 1 0.9 0.1; 1 0.7 0.6;] )
h = colorbar;
xticks([1 5 10 15 20 25 30 35 40])
xticklabels({'1979', '1983','1988','1993','1998','2003','2008','2013','2018'});
xtickangle(90);
xlabel('Years');
ylabel('Calendar Date');
set(h,'Ticks',[1.5 2.25 3 3.8 4.75 5.6 6.55],'TickLabels',["1" "2" "3" "4" "5" "6" "7"])
title('WTs per Year by Calendar Date');

%Plot the data
figure;hAxes = gca;
imagesc(hAxes,clustersnew);
colormap( hAxes , [ 0.4 0.7 1; 1 1 1;  1 1 1; 1 1 1; 0.6 0.8 1; 1 1 1; 1 1 1;] )
h = colorbar;
xticks([1 5 10 15 20 25 30 35 40])
xticklabels({'1979', '1983','1988','1993','1998','2003','2008','2013','2018'});
xtickangle(90);
xlabel('Years');
ylabel('Calendar Date');
set(h,'Ticks',[1.5 2.25 3 3.8 4.75 5.6 6.55],'TickLabels',["1" "2" "3" "4" "5" "6" "7"])

title('WTs per Year by Calendar Date');

%Plot the data
figure;hAxes = gca;
imagesc(hAxes,clustersnew);
colormap( hAxes , [ 1 1 1; 0.4 0.7 1; 1 1 1; 0.4 0.3 0.8; 1 1 1; 1 1 1; 1 1 1;] )
h = colorbar;
xticks([1 5 10 15 20 25 30 35 40])
xticklabels({'1979', '1983','1988','1993','1998','2003','2008','2013','2018'});
xtickangle(90);
xlabel('Years');
ylabel('Calendar Date');
set(h,'Ticks',[1.5 2.25 3 3.8 4.75 5.6 6.55],'TickLabels',["1" "2" "3" "4" "5" "6" "7"])
title('WTs per Year by Calendar Date');

%'TickLabels',{'Cold','Cool','Neutral','Warm','Hot'}