cl__1 = 1e+22;
Point(1) = {0.5, -0.5, 0, cl__1};
Point(2) = {-0.5, -0.5, 0, cl__1};
Point(3) = {-0.5, 0.5, 0, cl__1};
Point(4) = {0.5, 0.5, 0, cl__1};
Point(5) = {0, 0.7, 0, cl__1};
Point(6) = {0.7, -0, 0, cl__1};
Point(7) = {-0, -0.7, 0, cl__1};
Point(8) = {-0.7, -0, 0, cl__1};
Spline(1) = {3, 5, 4};
Spline(2) = {4, 6, 1};
Spline(3) = {1, 7, 2};
Spline(4) = {2, 8, 3};
Line Loop(1) = {1, 2, 3, 4};
Surface(1) = {1};
