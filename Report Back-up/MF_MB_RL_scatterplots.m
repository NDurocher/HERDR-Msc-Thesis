clear;
clc;
close all;


n = 700;

mu = [0.285; -0.54];
sigma = [0.03 0.03];
r = mvnrnd(mu,sigma,n);

b = [0, 1];
N = unidrnd(100,[n,2]);

N = N.*[1.5 2]./100 - b;

% subplot(2,1,2)
% scatter(N(:,1), N(:,2),'rx')
diev = std(r);

mu = mu + [0.2; 0.3];
opt_mean = mu;
distfromopt = vecnorm(N-ones(n,1)*opt_mean',2,2);
distfromopt = -distfromopt/max(distfromopt);
distfrommean = vecnorm(r-ones(n,1)*mu',2,2);
distfrommean = -distfrommean/max(distfrommean);
f = figure;
f.Position = [100 100 640 550];
map = [0.95 0 0; 0.9290 0.6940 0.1250; 0 .90 0];
colormap(map)


subplot(2,1,1)
scatter(r(:,1), r(:,2),[],distfrommean,'filled');
xlim([0.1 1.5])
ylim([-1. 1.])
title('Expected Reward Model-Free RL \pi(s_t,a_t)')
xlabel('a_0_t')
ylabel('a_1_t')
colorbar()


subplot(2,1,2)
scatter(N(:,1), N(:,2),[],distfromopt,'filled');
xlim([0.1 1.5])
ylim([-1. 1.])
title('Expected Reward Model-Based RL \pi(s_t,a_t)')
xlabel('a_0_t')
ylabel('a_1_t')
colorbar()

% s.AlphaData = distfromzero;
% s.MarkerFaceAlpha = 'flat';

% for i = 1:length(N)
%     
%     x = norm(diev-r(i,:));
%     if x<1.25
%       k = [0 1 0];
%     elseif (x>=1.25) && (x<2.25)
%       k = [0.9290 0.6940 0.1250];
%     elseif x>2.25
%       k = [1 0 0];
%     end
%     subplot(2,1,1)
%     scatter(r(i,1), r(i,2),'x', 'MarkerEdgeColor',k)
%     hold on
%     
%     y = norm(N(i,:)-opt_mean);
%     if y<1
%       c = [0 1 0];
%     elseif (y>=1) && (y<1.75)
%       c = [0.9290 0.6940 0.1250];
%     elseif y>1.75
%       c = [1 0 0];
%     end
%     subplot(2,1,2)
%     scatter(N(i,1), N(i,2),'x','MarkerEdgeColor',c)
%     hold on
% end

