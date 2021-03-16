AOactual = readmatrix("NAO_actual.csv");

total1 = sum(AOactual(1:3,:));
totalbot = sum(AOactual(4:6,:));
totaltop = sum(AOactual(7:9,:));

AOactual(1:3,:) = AOactual(1:3,:) ./ total1;
AOactual(4:6,:) = AOactual(4:6,:) ./ total1;
AOactual(7:9,:) = AOactual(7:9,:) ./ total1;

AOactual = AOactual * 100;

figure
hold on
for j=1:7
   hf=fill([j-.30 j-.01 j-.01 j-.30 j-.30],[AOactual(4,j) AOactual(4,j) AOactual(7,j) AOactual(7,j) AOactual(4,j)],rgb('grey'),'edgecolor','none');
hold on
   hf=fill([j+.30 j+.01 j+.01 j+.30 j+.30],[AOactual(6,j) AOactual(6,j) AOactual(9,j) AOactual(9,j) AOactual(6,j)],rgb('grey'),'edgecolor','none');

end
for j=1:7
        if AOactual(1,j)>AOactual(7,j) 
            cols='b';
        else if AOactual(1,j)<AOactual(4,j) 
                cols='r';
            else
                cols='k';
            end
        end
        if AOactual(3,j)>AOactual(9,j) 
            cols2='b';
        else if AOactual(3,j)<AOactual(6,j) 
                cols2='r';
            else
                cols2='k';
            end
        end
        h=bar(j,[AOactual(1,j), AOactual(3,j)]);
        h(1).FaceColor = cols;
h(2).FaceColor = cols2;

end

xlim([0 8])
ylim([0 50])
        xticklabels({' ','1','2','3','4','5','6','7',' '})
        xlabel("WT")
        ylabel("% of days")
        title('ENSO')
        %print(gcf,'ao_sonnew.png')

