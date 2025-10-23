S = csvread('S.csv', 1,1);
A = csvread('A.csv', 1,1);
B = csvread('B.csv', 1,1);
n = size(S,2);
N = size(S,1);
lam = 0.5; % Define the regularization parameter
cvx_begin
 variable x(N,n)
 minimize(-sum(log(sum(S.*x, 2))) + lam * sum(sum(abs(A * x),1),2))
 subject to
 x >= 0.001
 sum(x,2) == 1
cvx_end
y = cvx_optval;
% NEW METHOD
D = size(B,2);
K = size(B,1);
N = size(S,1);
for t=1:3
 SB = S*B;
 cvx_begin
 variable x(N,D)
 minimize(-sum(log(sum(SB.*x, 2))) + lam * sum(sum(abs(A * x),1),2))
 subject to
 x >= 0.001
 sum(x,2) == 1
 cvx_end
 y = cvx_optval;
 cvx_begin
 variable B(K,D)
 minimize(-sum(log(sum((S*B).*x, 2))))
 subject to
 B >= 0.001
 sum(B,1) == 1
 cvx_end
 y2 = cvx_optval+ lam * sum(sum(abs(A * x),1),2);
 disp(y2)
end
% FIX BUG
D = size(B,2);
K = size(B,1);
N = size(S,1);
M = size(A,1);
%cvx_precision('low') % default
for t=1:2
 cvx_begin
 variable x(N,D)
 minimize(-sum(log(sum(S.*(x*transpose(B)), 2))) + lam * sum(sum(abs(A * x),1),2))
 subject to
 x >= 0.001
 sum(x,2) == 1
 cvx_end
 y = cvx_optval;
 cvx_begin
 variable B(K,D)
 minimize(-sum(log(sum(S.*(x*transpose(B)), 2))))
 subject to
 B >= 0.001
 sum(B,1) == 1
 cvx_end
 y2 = cvx_optval+ lam * sum(sum(abs(A * x),1),2);
 disp(y2)
end


% ADD IN weights w - SCENARIO 3
Axs = sum(abs(A * x),2);
cvx_begin
 variable w(M)
 minimize(sum(w .* Axs,1))
 subject to
 w >= .01
 w <= 0.5
 abs(transpose(A)) * w == 1
cvx_end

% Save initial weights
csvwrite('cvx_weights_w.csv', w);
csvwrite('B_before_scenario3.csv', B);
csvwrite('X_before_scenario3.csv', x);
% Store objectives for scenario 3
objectives3 = zeros(3, 1);

for t=1:3
 cvx_begin
 variable x(N,D)
 minimize(-sum(log(sum(S.*(x*transpose(B)), 2))) + lam * sum(w .* sum(abs(A * x),2),1))
 subject to
 x >= 0.001
 sum(x,2) == 1
 cvx_end
 y = cvx_optval;
 cvx_begin
 variable B(K,D)
 minimize(-sum(log(sum(S.*(x*transpose(B)), 2))))
 subject to
 B >= 0.001
 sum(B,1) == 1
 cvx_end
 y2 = cvx_optval+ lam * sum(w .* sum(abs(A * x),2),1);
 disp(y2)


 
 Axs = sum(abs(A * x),2);
 cvx_begin
 variable w(M)
 minimize(sum(w .* Axs,1))
 subject to
 w >= .01
 w <= 0.5
 abs(transpose(A)) * w == 1
 cvx_end
 y2 = -sum(log(sum(S.*(x*transpose(B)), 2))) + lam * sum(w .* sum(abs(A * x),2),1);
 disp(y2)
 
 % Store objective
 objectives3(t) = y2;
end

% Save final results from Scenario 3
csvwrite('cvx_result3_X.csv', x);
csvwrite('cvx_result3_B.csv', B);
csvwrite('cvx_result3_obj.csv', objectives3);

fprintf('Scenario 3 results saved:\n');
fprintf('  cvx_result3_X.csv\n');
fprintf('  cvx_result3_B.csv\n');
fprintf('  cvx_result3_obj.csv\n');
fprintf('  cvx_weights_w.csv\n');