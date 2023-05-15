import numpy as np
import pandas as pd


class SyntheticData:
    
    def __init__(self):
        pass

    def model_1(self, n_samples):
    # dataset 1: linear probit model with no confounding
    # np.random.seed(101)
        def generate_model_1():
            h = -1
            n = 2
            I2 = np.identity(n)
            X = np.random.multivariate_normal([5, 5], I2)
            X0 = X[0]
            X1 = X[1]
            P = np.random.normal(5, 1)
            epsilon = np.random.normal(0, 1)
                
            # compute optimal price
            g = X0
            optimal_price = -(g + epsilon)/h

            Y_star = g + h*P + epsilon 

            return X0, X1, P, Y_star, optimal_price

        results = {
            'X0':[],
            'X1':[],
            'price':[],
            'optimal_price':[],
            'Y':[]
            }

        for _ in range(n_samples):
            X0, X1, P, Y_star, optimal_price = generate_model_1()
            results['X0'].append(X0)
            results['X1'].append(X1)
            results['price'].append(P)
            results['optimal_price'].append(optimal_price)
            results['Y'].append(int(Y_star > 0))

        results = pd.DataFrame(results)
        return results

    def model_2(self, n_samples):
        # dataset 2
        np.random.seed(101)
        def generate_model_2():
            g = 5
            X_prime = np.random.multivariate_normal([0]*20, np.identity(20))

            # draw beta 1 through 5
            beta_5 = np.random.multivariate_normal([0]*5, np.identity(5))
            beta_20 = [0]*15
            beta = np.concatenate([beta_5, beta_20])

            # take linear combo to get h(X)
            h = -1.5*(X_prime*beta).sum()

            # draw prices and noise
            P = np.random.normal(0, 2)
            epsilon = np.random.normal(0, 1)
            
            # compute optimal price
            optimal_price = -1*(g + epsilon)/h

            # compute Y*
            Y_star = g + h*P + epsilon 

            return X_prime, P, Y_star, optimal_price

        cols = ['X' + str(i) for i in range(20)] + ['price', 'optimal_price', 'Y']
        results = pd.DataFrame(columns=cols)

        for i in range(n_samples):
            X_prime, P, Y_star, optimal_price = generate_model_2()
            x = X_prime.tolist() + [P, optimal_price, int(Y_star > 0)]

            results.loc[i] = x

        return results

    def model_3(self, n_samples):
        # dataset 2
        np.random.seed(101)
        def generate_model_3():
            g = 5
            
            # first draw X0 and X1:
            n = 2
            I2 = np.identity(n)
            X = np.random.multivariate_normal([0, 0], I2)
            X0 = X[0]
            X1 = X[1]

            # next find piecewise-constant step function h
            if X0 < -1:
                h = -1.2
            elif -1 <= X0 < 0:
                h = -1.1
            elif 0 <= X0 < 1:
                h = -0.9
            elif 1 <= X0:
                h = -0.8

            # draw prices and noise
            P = np.random.normal(X0 + 5, 2)
            epsilon = np.random.normal(0, 1)

            # compute Y*
            Y_star = X0 + h*P + epsilon 

            # compute optimal price
            optimal_price = -1*(g + epsilon)/h

            return X0, X1, P, Y_star, optimal_price

        results = {
            'X0':[],
            'X1':[],
            'price':[],
            'Y':[],
            'optimal_price':[]
            }

        for i in range(n_samples):
            X0, X1, P, Y_star, optimal_price = generate_model_3()
            results['X0'].append(X0)
            results['X1'].append(X1)
            results['price'].append(P)
            results['Y'].append(int(Y_star > 0))
            results['optimal_price'].append(optimal_price)

        results = pd.DataFrame(results)

        return results

    def model_4(self, n_samples): 
        # dataset 2
        # np.random.seed(101)
        def generate_model_4():
            g = 5

            # first draw X0 and X1:
            n = 2
            I2 = np.identity(n)
            X = np.random.multivariate_normal([0, 0], I2)
            X0 = X[0]
            X1 = X[1]

            # next find piecewise-constant step function h
            # X1 influence price sensitivity, must be a 
            # consumer attribute
            h = -1.25 * int(X0 < -1) + \
            -1.1 * int(-1 <= X0 < 0) + \
            -0.9 * int(0 <= X0 < 1) + \
            -0.75 * int(1 <= X0) + \
            -0.1 * int(X1 < 0) + \
            0.1 * int(X1 >= 0)            # less sensitive to price

            # draw prices and noise
            # price depends on X0, so must be 
            # a product attribute
            P = np.random.normal(X0 + 5, 2)
            epsilon = np.random.normal(0, 1)

            # compute Y*
            Y_star = g + h*P + epsilon 

            optimal_price = -1*(g + epsilon)/h

            return X0, X1, P, Y_star, optimal_price

        results = {
            'X0':[],
            'X1':[],
            'price':[],
            'Y':[],
            'optimal_price':[]
                }

        for i in range(n_samples):
            X0, X1, P, Y_star, optimal_price = generate_model_4()
            results['X0'].append(X0)
            results['X1'].append(X1)
            results['price'].append(P)
            results['Y'].append(int(Y_star > 0))
            results['optimal_price'].append(optimal_price)

        results = pd.DataFrame(results)
        return results

    def model_5(self, n_samples):
        # dataset 2: linear probit model with no confounding
        np.random.seed(101)
        def generate_model_5():
            h = -1
            n = 2
            I2 = np.identity(n)
            X = np.random.multivariate_normal([0, 0], I2)
            X0 = X[0]
            X1 = X[1]
            P = np.random.normal(X0 + 5, 2)
            epsilon = np.random.normal(0, 1)

            Y_star = X0 + h*P + epsilon 

            optimal_price = -1*(X0 + epsilon)/h

            return X0, X1, P, Y_star, optimal_price

        results = {
            'X0':[],
            'X1':[],
            'price':[],
            'Y':[],
            'optimal_price':[]
                }

        for i in range(n_samples):
            X0, X1, P, Y_star, optimal_price = generate_model_5()
            results['X0'].append(X0)
            results['X1'].append(X1)
            results['price'].append(P)
            results['Y'].append(int(Y_star > 0))
            results['optimal_price'].append(optimal_price)

        results = pd.DataFrame(results)
        return results 

    def model_6(self, n_samples):
        # dataset 2: linear probit model with no confounding
        np.random.seed(101)
        def generate_model_6():
            # first draw X0 and X1:
            I2 = np.identity(2)
            X = np.random.multivariate_normal([0, 0], I2)
            X0 = X[0]
            X1 = X[1]

            # compute intercept and slope
            g = 4*np.abs(X0 + X1)
            h = -1*np.abs(X0 + X1)

            # draw price and noise
            P = np.random.normal(X0 + 5, 2)
            epsilon = np.random.normal(0, 1)

            # compute Y*
            Y_star = X0 + h*P + epsilon 

            # optimal price
            optimal_price = -1*(g + epsilon)/h

            return X0, X1, P, Y_star, optimal_price

        results = {
            'X0':[],
            'X1':[],
            'price':[],
            'Y':[],
            'optimal_price':[]
                }

        for i in range(n_samples):
            X0, X1, P, Y_star, optimal_price = generate_model_6()
            results['X0'].append(X0)
            results['X1'].append(X1)
            results['price'].append(P)
            results['Y'].append(int(Y_star > 0))
            results['optimal_price'].append(optimal_price)

        results = pd.DataFrame(results)
        return results

    def generate_data(self, model_id, n_samples):
        if model_id == 1:
            return self.model_1(n_samples)
        elif model_id == 2:
            return self.model_2(n_samples)
        elif model_id == 3:
            return self.model_3(n_samples)
        elif model_id == 4:
            return self.model_4(n_samples)
        elif model_id == 5:
            return self.model_5(n_samples)
        elif model_id == 6:
            return self.model_6(n_samples)
