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

%Find the original transition matrix
T = zeros(8,8);
C = zeros(8,8);
D = zeros(8,8);

for i = 1:40
    for j = 1:89
       T(knew(j,i),knew(j+1,i)) = T(knew(j,i),knew(j+1,i)) + 1; 
    end
end

%Now shuffle the days and repeat 10000 times
for i = 1:10000
   knew2 =K(:,7);
   knew2 =knew2(randperm(numel(knew2)));
   knew2 = reshape(knew2,91,[]);
   Ttemp = zeros(8,8);
   
   for ii = 1:40
       for jj = 1:89
           Ttemp(knew2(jj,ii),knew2(jj+1,ii)) = Ttemp(knew2(jj,ii),knew2(jj+1,ii)) + 1; 
       end
   end
   
   for l = 1:7
       for ll = 1:7
        valt = T(l,ll);
        valtemp = Ttemp(l,ll);
        
        if(valtemp <= valt)
            D(l,ll) = D(l,ll) + 1;
        end
        
        if(valtemp >= valt)
            C(l,ll) = C(l,ll) + 1;
        end
        
       end
   end
end

knew = K(:,5);
knew = reshape(knew,90,[]);
knew = knew(1:31,:);

%Find the original transition matrix
Tsep = zeros(5,5);
Csep = zeros(5,5);
Dsep = zeros(5,5);

for i = 1:40
    for j = 1:30
       Tsep(knew(j,i),knew(j+1,i)) = Tsep(knew(j,i),knew(j+1,i)) + 1; 
    end
end

%Now shuffle the days and repeat 10000 times
for i = 1:10000
   knew2 =K(:,5);
   knew2 =knew2(randperm(numel(knew2)));
   knew2 = reshape(knew2,90,[]);
   knew2 = knew2(1:31,:);
   Ttemp = zeros(5,5);
   
   for ii = 1:40
       for jj = 1:30
           Ttemp(knew2(jj,ii),knew2(jj+1,ii)) = Ttemp(knew2(jj,ii),knew2(jj+1,ii)) + 1; 
       end
   end
   
   for l = 1:5
       for ll = 1:5
        valt = Tsep(l,ll);
        valtemp = Ttemp(l,ll);
        
        if(valtemp <= valt)
            Dsep(l,ll) = Dsep(l,ll) + 1;
        end
        
        if(valtemp >= valt)
            Csep(l,ll) = Csep(l,ll) + 1;
        end
        
       end
   end
end

knew = K(:,5);
knew = reshape(knew,90,[]);
knew = knew(32:62,:);

%Find the original transition matrix
Toct = zeros(7,7);
Coct = zeros(7,7);
Doct = zeros(7,7);

for i = 1:40
    for j = 1:30
       Toct(knew(j,i),knew(j+1,i)) = Toct(knew(j,i),knew(j+1,i)) + 1; 
    end
end

%Now shuffle the days and repeat 10000 times
for i = 1:10000
   knew2 =K(:,7);
   knew2 =knew2(randperm(numel(knew2)));
   knew2 = reshape(knew2,91,[]);
   knew2 = knew2(31:61,:);
   Ttemp = zeros(7,7);
   
   for ii = 1:40
       for jj = 1:30
           Ttemp(knew2(jj,ii),knew2(jj+1,ii)) = Ttemp(knew2(jj,ii),knew2(jj+1,ii)) + 1; 
       end
   end
   
   for l = 1:7
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
Tnov = zeros(7,7);
Cnov = zeros(7,7);
Dnov = zeros(7,7);

for i = 1:40
    for j = 1:29
       Tnov(knew(j,i),knew(j+1,i)) = Tnov(knew(j,i),knew(j+1,i)) + 1; 
    end
end

%Now shuffle the days and repeat 10000 times
for i = 1:10000
   knew2 =K(:,7);
   knew2 =knew2(randperm(numel(knew2)));
   knew2 = reshape(knew2,91,[]);
   knew2 = knew2(62:91,:);
   Ttemp = zeros(7,7);
   
   for ii = 1:40
       for jj = 1:29
           Ttemp(knew2(jj,ii),knew2(jj+1,ii)) = Ttemp(knew2(jj,ii),knew2(jj+1,ii)) + 1; 
       end
   end
   
   for l = 1:7
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