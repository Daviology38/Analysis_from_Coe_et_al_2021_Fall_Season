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
oldK = K(:,7);
newK=[K(2:end,7);NaN]; %eliminate the last day
newK(mondays==1130)=NaN; %eliminate all Nov30 to Sep1 transitions

%% frequency, without caring about month
cnt=zeros(7,7);
newk=newK;
oldk=oldK;
for i=1:numel(oldk)
    if ~isnan(newk(i))
        cnt(oldk(i),newk(i))=cnt(oldk(i),newk(i))+1;
    end
end
pcnt=cnt./(sum(cnt,2)*ones(1,7))*100;	

%now do Monte Carlo of this
mpcnt=zeros(1000,7,7);
for m=1:1000
    cnt=zeros(7,7);
    oldk=oldK;
    newk=newK(randperm(numel(newK)));
    for i=1:numel(oldk)
        if ~isnan(newk(i))
            cnt(oldk(i),newk(i))=cnt(oldk(i),newk(i))+1;
        end
    end
    mpcnt(m,:,:)=cnt./(sum(cnt,2)*ones(1,7))*100;
end
lo=nan(7,7);
hi=nan(7,7);
for i=1:7
    for j=1:7
        tmp=mpcnt(:,i,j);
        tmpp = sort(tmp);
        lo(i,j)=tmpp(75);
        hi(i,j)=tmpp(925);
    end
end

figure
for i=1:7
    subplot_tight(4,2,i,[.08 .08]);
    hold on
    for j=1:7
    hf=fill([j-.5 j+.5 j+.5 j-.5 j-.5],[lo(i,j) lo(i,j) hi(i,j) hi(i,j) lo(i,j)],[.17 .17 .17],'edgecolor','none');
    end
    for j=1:7
        if pcnt(i,j)>hi(i,j) 
            cols='b';
        else if pcnt(i,j)<lo(i,j) 
                cols='r';
            else
                cols='k';
            end
        end
        h=bar(j,pcnt(i,j),cols);
    end
    xlim([0 8]);
    set(gca,'Xtick',1:7); 
    ylabel('Percent');
    ylim([0,75]);
    title(sprintf('WT%d',i));
end
print(gcf,'-dpng',sprintf('SON_transition85.png'));

% %% now do by month
% moncnt=zeros(3,7,7);
% for mm=1:3
%     f=mons==mm+8;
%     themons=mons(f);
%     oldk=oldK(f);
%     newk=newK(f);
%     for i=1:numel(oldk)
%         if ~isnan(newk(i))
%             moncnt(mm,oldk(i),newk(i))=moncnt(mm,oldk(i),newk(i))+1;
%         end
%     end
% end
% monpcnt=zeros(3,7,7);
% for mm=1:3
%     monpcnt(mm,:,:)=squeeze(moncnt(mm,:,:))./(sum(squeeze(moncnt(mm,:,:)),2)*ones(1,7))*100;	
% end
% 
% %monte carlo - just shuffle among all dates here
% mmonpcnt=zeros(1000,3,7,7);
% for m=1:1000
%     moncnt=zeros(3,7,7);
%     rK=newK(randperm(numel(newK)));
%     for mm=1:3
%         f=mons==mm+8;
%         themons=mons(f);
%         oldk=oldK(f);
%         newk=rK(f);        
%         for i=1:numel(oldk)
%             if ~isnan(newk(i))
%                 moncnt(mm,oldk(i),newk(i))=moncnt(mm,oldk(i),newk(i))+1;
%             end
%         end
%     end 
%     for mm=1:3
%         mmonpcnt(m,mm,:,:)=squeeze(moncnt(mm,:,:))./(sum(squeeze(moncnt(mm,:,:)),2)*ones(1,7))*100;	
%     end    
% end
% monlo=nan(3,7,7);
% monhi=nan(3,7,7);
% for mm=1:3
%     for i=1:7
%         for j=1:7
%             tmp=squeeze(mmonpcnt(:,mm,i,j));
%             monlo(mm,i,j)=tmp(25);
%             monhi(mm,i,j)=tmp(975);
%         end
%     end
% end
% 
% %figure
% montxt={'Sep','Oct','Nov'};
% for m=1:3
%     figure
%     for i=1:7
%         subplot_tight(4,2,i,[.08 .08]);
%         hold on
%        for j=1:7
%            hf=fill([j-.5 j+.5 j+.5 j-.5 j-.5],[monlo(m,i,j) monlo(m,i,j) monhi(m,i,j) monhi(m,i,j) monlo(m,i,j)],[.17 .17 .17],'edgecolor','none');
%        end
%         for j=1:7
%             if monpcnt(m,i,j)>monhi(m,i,j)
%                 cols='b';
%             else if monpcnt(m,i,j)<monlo(m,i,j)
%                     cols='r';
%                 else
%                     cols='k';
%                 end
%             end
%             h=bar(j,monpcnt(m,i,j),cols);
%         end
%         xlim([0 8]);
%         set(gca,'Xtick',1:7);
%         ylabel('Percent');
%         ylim([0,75]);
%         title(sprintf('WT%d',i));
%         sgtitle(sprintf('%s',montxt{m}));
%     end
%     print(gcf,'-dpng',sprintf('%s_transition.png',montxt{m}));
% end
% 
% %% by month again, but this time shuffle clusters within months only
% moncnt=zeros(3,7,7);
% for mm=1:3
%     f=mons==mm+8;
%     themons=mons(f);
%     oldk=oldK(f);
%     newk=newK(f);
%     for i=1:numel(oldk)
%         if ~isnan(newk(i))
%             moncnt(mm,oldk(i),newk(i))=moncnt(mm,oldk(i),newk(i))+1;
%         end
%     end
% end
% monpcnt=zeros(3,7,7);
% for mm=1:3
%     monpcnt(mm,:,:)=squeeze(moncnt(mm,:,:))./(sum(squeeze(moncnt(mm,:,:)),2)*ones(1,7))*100;	
% end
% 
% %monte carlo - shuffle transitions WITHIN month
% mmonpcnt=zeros(1000,3,7,7);
% for m=1:1000
%     moncnt=zeros(3,7,7);
%     for mm=1:3
%         f=mons==mm+8;
%         themons=mons(f);
%         oldk=oldK(f);
%         newk=newK(f); 
%         rk=newk(randperm(numel(newk)));
%         newk=rk; %just assign to the shuffled deck
%         for i=1:numel(oldk)
%             if ~isnan(newk(i))
%                 moncnt(mm,oldk(i),newk(i))=moncnt(mm,oldk(i),newk(i))+1;
%             end
%         end
%     end 
%     for mm=1:3
%         mmonpcnt(m,mm,:,:)=squeeze(moncnt(mm,:,:))./(sum(squeeze(moncnt(mm,:,:)),2)*ones(1,7))*100;	
%     end    
% end
% monlo=nan(3,7,7);
% monhi=nan(3,7,7);
% for mm=1:3
%     for i=1:7
%         for j=1:7
%             tmp=squeeze(mmonpcnt(:,mm,i,j));
%             monlo(mm,i,j)=tmp(25);
%             monhi(mm,i,j)=tmp(975);
%         end
%     end
% end
% 
% %figure
% montxt={'Sep','Oct','Nov'};
% for m=1:3
%     figure
%     for i=1:7
%         subplot_tight(4,2,i,[.08 .08]);
%         hold on
%        for j=1:7
%            hf=fill([j-.5 j+.5 j+.5 j-.5 j-.5],[monlo(m,i,j) monlo(m,i,j) monhi(m,i,j) monhi(m,i,j) monlo(m,i,j)],[.17 .17 .17],'edgecolor','none');
%        end
%         for j=1:7
%             if monpcnt(m,i,j)>monhi(m,i,j)
%                 cols='b';
%             else if monpcnt(m,i,j)<monlo(m,i,j)
%                     cols='r';
%                 else
%                     cols='k';
%                 end
%             end
%             h=bar(j,monpcnt(m,i,j),cols);
%         end
%         xlim([0 8]);
%         set(gca,'Xtick',1:7);
%         ylabel('Percent');
%         ylim([0,75]);
%         title(sprintf('WT%d',i));
%     end
%     print(gcf,'-dpng',sprintf('%s_transition2.png',montxt{m}));
% end