import numpy as np
from StudentPrescriptionTree import StudentPrescriptionTree
from SyntheticData import SyntheticData
from RiskControlledSPT import RiskControlledSPT


class EvaluateSPT:
    def __init__(self, X, y, price_thresh, tree_model, purchase_prob=None):
        self.X = X
        self.y = y
        self.price_thresh = price_thresh
        self.true_revenue_matrix = None
        self.tree1 = None
        self.tree2 = None
        self.purchase_prob = purchase_prob
        self.SyntheticData = SyntheticData()
        self.tree_model = tree_model

    def get_true_revenue_matrix(self, prices):
        """
        Get revenue matrix corresponding to ground truth
        generative probability distribution
        X: data
        prices: descretized prices
        """
        # make empty response matrix
        revenue_matrix = np.empty((len(self.X), len(prices)))

        for i, price in enumerate(prices):
            # construct reponse matrix
            # price threshold used to determine
            # sale outcome
            revenue_matrix[:, i] = (price < self.price_thresh) * price

        self.true_revenue_matrix = revenue_matrix

    def get_pred_revenue_matrix(self, prices, X_test, model_id):
        """
        Get revenue matrix corresponding to ground truth
        generative probability distribution
        X: data
        prices: descretized prices
        """
        # make empty response matrix
        revenue_matrix = np.empty((len(self.X), len(prices)))

        for i, price in enumerate(prices):
            # construct reponse matrix
            # price threshold used to determine
            # sale outcome
            Y_star = self.SyntheticData.get_purchase_prob(
                model_id=model_id,
                price=price,
                X_test=X_test
            )
            revenue_matrix[:, i] = price * (1 / (1 + np.exp(-Y_star)))

        self.pred_revenue_matrix = revenue_matrix

    def fit_trees(self, model, max_depth, model_id=None):
        # fit training data
        tree1 = StudentPrescriptionTree(teacher_model=model, max_depth=max_depth)
        covariates = [x for x in self.X.columns if 'X' in x] + ['price']
        tree1.fit(self.X[covariates])

        # get ground truth revenue matrix
        # prices are descretized from 0.1 to 0.9 percentiles
        # of observed prices in sample, cut into 10 prices 
        self.get_true_revenue_matrix(tree1.prices)
        self.get_pred_revenue_matrix(tree1.prices, self.X, model_id)

        # train using ground truth
        tree2 = StudentPrescriptionTree(teacher_model=model, max_depth=max_depth)
        tree2.fit(
            self.X[covariates], 
            revenue_matrix=self.true_revenue_matrix #self.pred_revenue_matrix
        )

        self.tree1 = tree1
        self.tree2 = tree2

    def evaluateSPT(self, depth=None, X_test=None):

        def get_rev(tree, depth):

            revs = []
            # get leave nodes for tree
            if X_test is None:
                leaf_nodes = tree.get_nodes(depth=depth)
            else:
                leaf_nodes = tree.get_test_nodes(X_test, max_depth=depth)

            for leaf_node in leaf_nodes:
                # get price index
                k = leaf_node['price']
                # get datapoints
                S = leaf_node['datapoints']

                # we can then reference the underlying
                # generative model to determine if the 
                # user purchases or does not
                rev = self.true_revenue_matrix[S, k].sum()

                #total_revenue.append(partition_rev)
                revs.append(rev)

            return revs

        opt_revenue1 = get_rev(self.tree1, depth)
        opt_revenue2 = get_rev(self.tree2, depth)

        spt_revenue = sum(opt_revenue1) / len(self.X)
        opt_revenue = sum(opt_revenue2) / len(self.X)

        return spt_revenue, opt_revenue


