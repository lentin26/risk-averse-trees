import numpy as np
from StudentPrescriptionTree import StudentPrescriptionTree


class EvaluateSPT:
    def __init__(self, X, y, price_thresh):
        self.X = X
        self.y = y
        self.price_thresh = price_thresh
        self.true_revenue_matrix = None
        self.tree1 = None
        self.tree2 = None

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

    def fit_trees(self, model, max_depth):
        # fit training data
        tree1 = StudentPrescriptionTree(teacher_model=model, max_depth=max_depth)
        tree1.fit(self.X)

        # get ground truth revenue matrix
        self.get_true_revenue_matrix(tree1.prices)

        # train using ground truth
        tree2 = StudentPrescriptionTree(teacher_model=model, max_depth=max_depth)
        tree2.fit(
            self.X, 
            revenue_matrix=self.true_revenue_matrix
        )

        self.tree1 = tree1
        self.tree2 = tree2

    def evaluateSPT(self, depth=None, X_test=None):

        def get_rev(tree, depth):
            revs = []

            # get descretized prices 
            # should be the same for both trees
            # since both trees were trained on the 
            # same data
            prices = tree.prices

            # get leave nodes for tree
            if X_test is None:
                leaf_nodes = tree.get_nodes(depth=depth)
            else:
                leaf_nodes = tree.predict(X_test.drop('optimal_price', axis=1))

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


