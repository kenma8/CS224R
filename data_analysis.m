% DAgger iterations
iterations = 0:9;

% Policy 1 results
mean_returns_1 = [4811, 4756, 4707, 4781, 4752, 4812, 4786, 3387, 4729, 4677];
std_returns_1  = [66, 76, 122, 103, 65, 186, 83, 1850, 29, 83];

% Policy 2 results
mean_returns_2 = [982, 1066, 3733, 3715, 3721, 3721, 3717, 3720, 3715, 3714];
std_returns_2  = [135, 65, 6.8, 4.2,  2.1, 2.4, 3.2, 4.3, 2.0, 5.6];

% Your existing figure setup
figure;
hold on;

% Your error bar plots
errorbar(iterations, mean_returns_1, std_returns_1, '-o', 'LineWidth', 1.5, 'Color', 'red');
errorbar(iterations, mean_returns_2, std_returns_2, '-o', 'LineWidth', 1.5, 'Color', 'blue');

% Add horizontal dotted lines for expert policies
yline(4714, '--', 'Ant Expert Policy', 'Color', 'red', ...
      'LineWidth', 1.5, 'LabelHorizontalAlignment', 'left');

yline(3773, '--', 'Hopper Expert Policy', 'Color', 'blue', ...
      'LineWidth', 1.5, 'LabelHorizontalAlignment', 'left');

% Label and format
xlabel('DAgger Iteration');
ylabel('Mean Reward');
title('Policy Performance over DAgger Iterations');
legend('Ant Environment', 'Hopper Environment', 'Location', 'southeast');
grid on;
