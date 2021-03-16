%this gets the transitions for WTs

%close all; clear variables;

%full set of dates
sdate='19790101'; snum=datenum(sdate,'yyyymmdd');
edate='20181231'; enum=datenum(edate,'yyyymmdd');
dates=str2num(datestr(snum:enum,'yyyymmdd'));
mons=int32(mod(dates,1e4)/1e2);

%winnow down to Sep-Nov
dates=dates(mons==9|mons==10|mons==11);
mons=floor(mod(dates,1e4)/1e2); %recalc
years=int32(dates/1e4);
mondays=int32(floor(mod(dates,1e4)));

%get K assignments
%tmp=csvread('COE_SONclusts.csv',1,0);
%K=tmp(:,2);
%oldK=K(:,6);
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

knew = K(:,7);
knew = reshape(knew,91,[]);

total_WTs = zeros(7);
for i = 1:3640
    total_WTs(K(i,7)) = total_WTs(K(i,7)) + 1;
end

%Find the original transition matrix
T = zeros(49,7);
C = zeros(49,7);
D = zeros(49,7);

for i = 1:40
    for j = 1:89
       v1 = knew(j,i);
       v2 = knew(j+1,i);
       if(v1 == 1)
           start = 1;
       elseif(v1 == 2)
           start = 8;
       elseif(v1 == 3)
           start = 15;
       elseif(v1 == 4)
           start = 22;
       elseif(v1 == 5)
           start = 29;
       elseif(v1 == 6)
           start = 36;
       else
           start = 43;
       end
       T(start+v2-1,knew(j+2,i)) = T(start+v2-1,knew(j+2,i)) + 1; 
    end
end

%Now repeat 10000 times, be sure to keep the number of days for each
%pattern the same for each run (using T).
total_row = sum(T,2);
for i = 1:10000
   knew2 =K(:,7);
   knew2 =knew2(randperm(numel(knew2)));
   
   count1 = 1;
   count2 = 1;
   for aa = 1:49
   
       
       temp_total = total_WTs;
       temp_total(count1) = temp_total(count1);
       temp_total(count2) = temp_total(count2);
       
       %Make array of size of total amounts of WTs
       wt_temp = zeros(sum(temp_total(:)),1);
       
       %Now fill the array with the numbers, dont worry about order as
       %the array will be randomly shuffled
       counting = 1;
       temporary = 1;
       for q = 1:length(wt_temp)
           
           wt_temp(q) = temporary;
           counting = counting + 1;
           if(counting > temp_total(temporary))
              counting = 1;
              temporary = temporary + 1;
           end
       end
       
       wt_temp = wt_temp(randperm(numel(wt_temp)));
       
       test_vals = datasample(wt_temp,total_row(aa),'Replace',false);
       new_vals = zeros(7);
       for nn = 1:length(test_vals)
          new_vals(test_vals(nn)) = new_vals(test_vals(nn)) + 1; 
       end
       
        for ll = 1:7
        valt = T(aa,ll);
        valtemp = new_vals(ll);
        
        if(valtemp <= valt)
            D(aa,ll) = D(aa,ll) + 1;
        end
        
        if(valtemp >= valt)
            C(aa,ll) = C(aa,ll) + 1;
        end
        
        end
       count2 = count2 + 1;
       if(mod(aa,7) == 0)
           count1 = count1 + 1;
           count2 = 1;
       end
       
       
   end
   
end

knew = K(:,7);
knew = reshape(knew,91,[]);
knew = knew(1:30,:);

%Find the original transition matrix
Tsep = zeros(49,7);
Csep = zeros(49,7);
Dsep = zeros(49,7);

for i = 1:40
    for j = 1:28
        v1 = knew(j,i);
       v2 = knew(j+1,i);
       if(v1 == 1)
           start = 1;
       elseif(v1 == 2)
           start = 8;
       elseif(v1 == 3)
           start = 15;
       elseif(v1 == 4)
           start = 22;
       elseif(v1 == 5)
           start = 29;
       elseif(v1 == 6)
           start = 36;
       else
           start = 43;
       end
       Tsep(start+v2-1,knew(j+2,i)) = Tsep(start+v2-1,knew(j+2,i)) + 1; 
    end
end
total_WTssep = zeros(7,1);
knews = knew(:);
for i = 1:length(knew(:))
   total_WTssep(knews(i)) = total_WTssep(knews(i)) + 1;
end
total_row = sum(Tsep,2);
%Now shuffle the days and repeat 10000 times
for i = 1:10000
   count1 = 1;
   count2 = 1;
    
   %Make array of size of total amounts of WTs
       wt_temp = zeros(sum(temp_total(:)),1);
       
       %Now fill the array with the numbers, dont worry about order as
       %the array will be randomly shuffled
       counting = 1;
       temporary = 1;
       for q = 1:length(wt_temp)
           
           wt_temp(q) = temporary;
           counting = counting + 1;
           if(counting > temp_total(temporary))
              counting = 1;
              temporary = temporary + 1;
           end
       end
       
       wt_temp = wt_temp(randperm(numel(wt_temp)));
       
       
   for aa = 1:49
      test_vals = datasample(wt_temp,total_row(aa),'Replace',false);
       temp_total = total_WTssep;
       temp_total(count1) = temp_total(count1);
       temp_total(count2) = temp_total(count2);
       
       
       new_vals = zeros(7);
       for nn = 1:length(test_vals)
          new_vals(test_vals(nn)) = new_vals(test_vals(nn)) + 1; 
       end
       
        for ll = 1:7
        valt = Tsep(aa,ll);
        valtemp = new_vals(ll);
        
        if(valtemp <= valt)
            Dsep(aa,ll) = Dsep(aa,ll) + 1;
        end
        
        if(valtemp >= valt)
            Csep(aa,ll) = Csep(aa,ll) + 1;
        end
        
        end
       count2 = count2 + 1;
       if(mod(aa,7) == 0)
          count1 = count1 + 1;
          count2 = 1;
       end
       
       
   end
end

knew = K(:,7);
knew = reshape(knew,91,[]);
knew = knew(31:61,:);

%Find the original transition matrix
Toct = zeros(49,7);
Coct = zeros(49,7);
Doct = zeros(49,7);

for i = 1:40
    for j = 1:29
        v1 = knew(j,i);
       v2 = knew(j+1,i);
       if(v1 == 1)
           start = 1;
       elseif(v1 == 2)
           start = 8;
       elseif(v1 == 3)
           start = 15;
       elseif(v1 == 4)
           start = 22;
       elseif(v1 == 5)
           start = 29;
       elseif(v1 == 6)
           start = 36;
       else
           start = 43;
       end
       Toct(start+v2-1,knew(j+2,i)) = Toct(start+v2-1,knew(j+2,i)) + 1; 
    end
end

%Now shuffle the days and repeat 10000 times
for i = 1:10000
   knew2 =K(:,7);
   knew2 =knew2(randperm(numel(knew2)));
   knew2 = reshape(knew2,91,[]);
   knew2 = knew2(31:61,:);
   Ttemp = zeros(49,7);
   
   for ii = 1:40
       for jj = 1:29
           v1 = knew2(jj,ii);
       v2 = knew2(jj+1,ii);
       if(v1 == 1)
           start = 1;
       elseif(v1 == 2)
           start = 8;
       elseif(v1 == 3)
           start = 15;
       elseif(v1 == 4)
           start = 22;
       elseif(v1 == 5)
           start = 29;
       elseif(v1 == 6)
           start = 36;
       else
           start = 43;
       end
           Ttemp(start+v2-1,knew2(jj+2,ii)) = Ttemp(start+v2-1,knew2(jj+2,ii)) + 1; 
       end
   end
   
   for l = 1:49
       for ll = 1:7
        valt = Toct(l,ll);
        valtemp = Ttemp(l,ll);
        
        if(valtemp <= valt)
            Doct(l,ll) = Doct(l,ll) + 1;
        end
        
        if(valtemp >= valt)
            Coct(l,ll) = Coct(l,ll) + 1;
        end
        
       end
   end
end


knew = K(:,7);
knew = reshape(knew,91,[]);
knew = knew(62:91,:);

%Find the original transition matrix
Tnov = zeros(49,7);
Cnov = zeros(49,7);
Dnov = zeros(49,7);

for i = 1:40
    for j = 1:28
        v1 = knew(j,i);
       v2 = knew(j+1,i);
       if(v1 == 1)
           start = 1;
       elseif(v1 == 1)
           start = 8;
       elseif(v1 == 3)
           start = 15;
       elseif(v1 == 4)
           start = 22;
       elseif(v1 == 5)
           start = 29;
       elseif(v1 == 6)
           start = 36;
       else
           start = 43;
       end
       Tnov(start+v2-1,knew(j+2,i)) = Tnov(start+v2-1,knew(j+2,i)) + 1; 
    end
end

%Now shuffle the days and repeat 10000 times
for i = 1:10000
   knew2 =K(:,7);
   knew2 =knew2(randperm(numel(knew2)));
   knew2 = reshape(knew2,91,[]);
   knew2 = knew2(62:91,:);
   Ttemp = zeros(49,7);
   
   for ii = 1:40
       for jj = 1:28
           v1 = knew2(jj,ii);
       v2 = knew2(jj+1,ii);
       if(v1 == 1)
           start = 1;
       elseif(v1 == 2)
           start = 8;
       elseif(v1 == 3)
           start = 15;
       elseif(v1 == 4)
           start = 22;
       elseif(v1 == 5)
           start = 29;
       elseif(v1 == 6)
           start = 36;
       else
           start = 43;
       end
           Ttemp(start+v2-1,knew2(jj+2,ii)) = Ttemp(start+v2-1,knew2(jj+2,ii)) + 1; 
       end
   end
   
   for l = 1:49
       for ll = 1:7
        valt = Tnov(l,ll);
        valtemp = Ttemp(l,ll);
        
        if(valtemp <= valt)
            Dnov(l,ll) = Dnov(l,ll) + 1;
        end
        
        if(valtemp >= valt)
            Cnov(l,ll) = Cnov(l,ll) + 1;
        end
        
       end
   end
end