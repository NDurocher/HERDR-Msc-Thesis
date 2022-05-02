clear;
clc;
close all;

r = rectangle('Position',[-5 2 10 10]);
r.LineWidth = 1.5;
r.FaceColor = [245, 245, 245]/255;
axis([-10 10 -1 15]);
grid on
title("Generalized Action Space")
xlabel("Steering Angle \delta")
ylabel("Linear Speed \nu")


text(1.0, 12.5, '\nu_{max}','FontSize',12)
text(1.0, 2.5, '\nu_{min}','FontSize',12)
text(-5.2, 1.5, '\delta_{min}','FontSize',12)
text(4.9, 1.5, '\delta_{max}','FontSize',12)

ax = gca;

ax.LabelFontSizeMultiplier = 1.2;
ax.TitleFontSizeMultiplier = 1.5;
ax.YAxisLocation = 'origin';
ax.XAxisLocation = 'origin';
ax.YAxis.LineWidth = 2;
ax.XAxis.LineWidth = 2;
ax.TickDir = 'both';
ax.XTick = 15:30;
ax.YTick = 16:30;
a = annotation('arrow',[0.5180, 0.5180],[0.2,0.9305]);
a.LineWidth = 2;
annotation('arrow',[0.5185, 0.9095],[0.161,0.161])