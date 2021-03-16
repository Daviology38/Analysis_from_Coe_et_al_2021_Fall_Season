%tmp = load('transition_matrix_son.mat');
%tm = tmp.transition;
% tm = transition_3;
% tm = [tm, zeros(49,42)];
tm = transpose(Tsep);

mc = dtmc(tm);

figure; 
h = graphplot(mc,'ColorEdges',true);
G = digraph(mc.P);
bins = conncomp(G);
p.MarkerSize = 7;
p.NodeCData = bins;
colormap(hsv(4))

%C = condensation(G);
%p2 = plot(C);
%p2.MarkerSize = 7;
%colormap(hsv(4))