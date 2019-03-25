import operator
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

#seed the random in case of needing reproduction
np.random.seed(21)
# Pruned BFS data x - n queen, y - time taken
x = np.array([1,2,3,4,5,6,7,8,9,10,11])
y = np.array([0.0, 0.0, 0.0, 0.0, 0.003004, 0.026025, 0.232299, 2.455049, 28.986122, 356.476245, 4574.794219])

x = x[:, np.newaxis]
y = y[:, np.newaxis]
# prediction of N = 30 queens
x_pred = np.array([30])
x_pred = x_pred[:, np.newaxis]

# linear regression
learn_lin = LinearRegression()
learn_lin.fit(x, y)
y_hat_lin = learn_lin.predict(x)
y_30_lin = learn_lin.predict(x_pred)

print("Linear prediction %f " % y_30_lin)
print("Linear Root Mean Squared Error: %f" % np.sqrt(mean_squared_error(y, y_hat_lin)))
print("Linear R_squared (determination): %f" % r2_score(y, y_hat_lin))

# polynomial regression
polynomial_features= PolynomialFeatures(degree=7)
x_poly = polynomial_features.fit_transform(x)
y_poly = polynomial_features.fit_transform(y)
x_pred_poly = polynomial_features.fit_transform(x_pred)

learn_poly = LinearRegression()
learn_poly.fit(x_poly, y)
y_hat_poly = learn_poly.predict(x_poly)
y_30_poly = learn_poly.predict(x_pred_poly)

print("Polynomial prediction %f " % y_30_poly)
print("Polynomial Root Mean Squared Error: %f" % np.sqrt(mean_squared_error(y, y_hat_poly)))
print("Polynomial R_squared (determination): %f" % r2_score(y, y_hat_poly))

x = np.array([1,2,3,4,5,6,7,8,9,10,11])
y = np.array([0.0, 0.0, 0.0, 0.0, 0.003004, 0.026025, 0.232299, 2.455049, 28.986122, 356.476245, 4574.794219])

# y = a*b^x + c exponential graph
def exp_function(x, a, b, c):
    return a * np.exp(b * x) + c

exp_popt, exp_pcov = curve_fit(exp_function, x, y)
y_hat_exp = exp_function(x, *exp_popt)
y_30_exp = exp_function(30, *exp_popt)

print("Exponential prediction %f " % y_30_exp)
print("Exponential Root Mean Squared Error: %f" % np.sqrt(mean_squared_error(y, y_hat_exp)))
print("Exponential R_squared (determination): %f" % r2_score(y, y_hat_exp))

# graph scatter and three different plots with labels and legends
plt.scatter(x,y, s=10)
linear, polynomial, exponential = plt.plot(x, y_hat_lin, 'y', x, y_hat_poly, 'r', x, exp_function(x, *exp_popt), 'c')
plt.xlabel('N-queen')
plt.ylabel('Time in seconds')
plt.legend((linear, polynomial, exponential), ('Linear', 'Polynomial', 'Exponential'))
plt.show()

