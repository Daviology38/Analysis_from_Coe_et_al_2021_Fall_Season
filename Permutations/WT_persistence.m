%This script finds the persistence per year of each WT

%Load in the data
data_temp = csvread('Coe_clusters_new.csv');
clusters = data_temp(2:end,2);

%Reshape to 91 dates per year for all years
clustersnew = reshape(clusters,91,[]);

%Arrays to hold the data
wt1 = zeros(40,20);
wt2 = zeros(40,20);
wt3 = zeros(40,20);
wt4 = zeros(40,20);
wt5 = zeros(40,20);
wt6 = zeros(40,20);
wt7 = zeros(40,20);

%Fill the arrays
for i = 1:40
    j = 1;
    counter = 1;
    while j < 91
        val1 = clustersnew(j,i);
        val2 = clustersnew(j+1,i);
        if(val1 == val2)
           counter = counter + 1;
           j = j + 1;
        else
           if(val1 == 1)
               wt1(i,counter) = wt1(i,counter) + 1;
           elseif(val1 == 2)
               wt2(i,counter) = wt2(i,counter) + 1;
           elseif(val1 == 3)
               wt3(i,counter) = wt3(i,counter) + 1;
           elseif(val1 == 4)
               wt4(i,counter) = wt4(i,counter) + 1;
           elseif(val1 == 5)
               wt5(i,counter) = wt5(i,counter) + 1;
           elseif(val1 == 6)
               wt6(i,counter) = wt6(i,counter) + 1;
           else
               wt7(i,counter) = wt7(i,counter) + 1;
           end
           j = j + 1;
           counter = 1;
        end
    end
    
end

for i = 1:40
   wt1(i,:) = wt1(i,:) / sum(wt1(i,:)); 
   wt2(i,:) = wt2(i,:) / sum(wt2(i,:)); 
   wt3(i,:) = wt3(i,:) / sum(wt3(i,:)); 
   wt4(i,:) = wt4(i,:) / sum(wt4(i,:)); 
   wt5(i,:) = wt5(i,:) / sum(wt5(i,:)); 
   wt6(i,:) = wt6(i,:) / sum(wt6(i,:)); 
   wt7(i,:) = wt7(i,:) / sum(wt7(i,:)); 
    
end