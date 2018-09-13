function X = Heaviside(X)
X = sign(X);
X(X<=0) = 0;