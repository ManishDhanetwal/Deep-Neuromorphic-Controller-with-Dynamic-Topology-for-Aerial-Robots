close all
% load('circle_withdist15102019.mat')
% batchPath_plot = "circlewithdist";

% load('circle_nodist12102019.mat')
% batchPath_plot = "circlenodist";

%load('circle_static_nolearn_weighthover21102019.mat')
%load('staticnet_circular_fixed_2021.mat')
%load('mocap_run_iris_velos_netevo_ceiling_with_initial_learning_gobelow_2021.mat')
%load('hovering1node2layers_fied.mat')
load('experiment Ceiling Static with Learning go to 1.3-0.5 below the ceiling weigh_20191029.mat')
batchPath_plot = "weightnodist";

golden_ratio = 1.618;
x0 = 1;
y0 = 1;
width = 1.95; %Width in terms of the format (e.g., 3.5).
height= width/golden_ratio;
text_font = 10;

%% XY Plot
figure('Units','inches',...
'Position',[x0 y0 (x0+width) (y0+height)],...
'PaperPositionMode','auto');

plot(Flight(:,3),Flight(:,4),'--');
hold on
plot(Flight(:,6),Flight(:,7));
grid on
xlabel('X (m)');
ylabel('Y (m)');
axis equal;
legend('Reference','UAV Trajectory');
hold off

set(gca,...
'Units','normalized',...
'FontUnits','points',...
'FontWeight','normal',...
'FontSize',text_font,...
'FontName','Times');
set(gca,'LooseInset',max(get(gca,'TightInset'), 0.02))
set(gcf,'renderer','Painters')
print((batchPath_plot+'XY.eps'),'-depsc');

%% X Control Signal Plot
figure('Units','inches',...
'Position',[x0 y0 (x0+width) (y0+height)],...
'PaperPositionMode','auto');

plot(Flight(:,1),X(:,2),'--','LineWidth',1.5);
hold on
plot(Flight(:,1),X(:,3),'LineWidth',1.5);
grid on
xlabel('Time(s)');
ylabel('X-Control signal');
legend('SMC','NN');
hold off

set(gca,...
'Units','normalized',...
'FontUnits','points',...
'FontWeight','normal',...
'FontSize',text_font,...
'FontName','Times');
set(gca,'LooseInset',max(get(gca,'TightInset'), 0.02))
set(gcf,'renderer','Painters')
print((batchPath_plot+'Xcon.eps'),'-depsc');

%% Y Control Signal Plot
figure('Units','inches',...
'Position',[x0 y0 (x0+width) (y0+height)],...
'PaperPositionMode','auto');

plot(Flight(:,1),Y(:,2),'--','LineWidth',1.5);
hold on
plot(Flight(:,1),Y(:,3),'LineWidth',1.5);
grid on
xlabel('Time(s)');
ylabel('Y-Control signal');
legend('SMC','NN');
hold off

set(gca,...
'Units','normalized',...
'FontUnits','points',...
'FontWeight','normal',...
'FontSize',text_font,...
'FontName','Times');
set(gca,'LooseInset',max(get(gca,'TightInset'), 0.02))
set(gcf,'renderer','Painters')
print((batchPath_plot+'Ycon.eps'),'-depsc');

%% X Control Evolution
figure('Units','inches',...
'Position',[x0 y0 (x0+width) (y0+height)],...
'PaperPositionMode','auto');

plot(Flight(:,1),X(:,4),'--','LineWidth',1.5);
hold on
plot(Flight(:,1),X(:,5),'LineWidth',1.5);
grid on
xlabel('Time(s)');
ylabel('X-Controller Evolution');
legend('Layer','Node');
hold off

set(gca,...
'Units','normalized',...
'FontUnits','points',...
'FontWeight','normal',...
'FontSize',text_font,...
'FontName','Times');
set(gca,'LooseInset',max(get(gca,'TightInset'), 0.02))
set(gcf,'renderer','Painters')
print((batchPath_plot+'Xevo.eps'),'-depsc');

%% Y Control Evolution
figure('Units','inches',...
'Position',[x0 y0 (x0+width) (y0+height)],...
'PaperPositionMode','auto');

plot(Flight(:,1),Y(:,4),'--','LineWidth',1.5);
hold on
plot(Flight(:,1),Y(:,5),'LineWidth',1.5);
grid on
xlabel('Time(s)');
ylabel('Y-Controller Evolution');
legend('Layer','Node');
hold off

set(gca,...
'Units','normalized',...
'FontUnits','points',...
'FontWeight','normal',...
'FontSize',text_font,...
'FontName','Times');

 
set(gca,'LooseInset',max(get(gca,'TightInset'), 0.02))
set(gcf,'renderer','Painters')
print((batchPath_plot+'Yevo.eps'),'-depsc');


