import numpy as np
import warnings


class RiskControlledSPT:
    def __init__(self, teacher_model, max_depth, upper_confidence, lower_confidence):
        self.teacher_model = teacher_model
        self.max_depth = max_depth
        self.root_node = None
        self.prices = None
        self.train_size = None
        self.upper_confidence = upper_confidence
        self.lower_confidence = lower_confidence
        self.pred_revenue_matrix = None

    def is_risk_controlled(self, proposed_rev, current_rev):
        x = proposed_rev - current_rev
        if sum(x) != 0:
            a = sum(x[x > 0]) / sum(x) >= self.upper_confidence
            b = sum(x[x < 0]) / sum(x) <= self.lower_confidence
            return a & b
        else:
            return False

    def get_best_price_and_revenue(self, datapoints):
        current_rev = 0
        best_price = None
        best_R = None
        for price in self.prices:
            proposed_rev = self.teacher_model.get_revenue_pred(price, datapoints)
            if self.is_risk_controlled(proposed_rev, current_rev):
                best_price = price
                best_R = proposed_rev

        return best_price, best_R

    def get_revenue_matrix(self, X):
        # get descrete prices
        prices = self.prices

        # make empty response matrix
        revenue_matrix = np.empty((
            len(self.teacher_model.alpha),
            len(X), 
            len(prices), 
        ))

        for i in range(len(prices)):
            price = prices[i]

            # construct reponse matrix
            revenue_matrix[:, :, i] = self.teacher_model.get_revenue_pred(price, X)

        self.pred_revenue_matrix = revenue_matrix

    def discretize_prices_(self, prices):
        # discretize prices
        low_price = prices.quantile(0.10)
        high_price = prices.quantile(0.90)

        self.prices = np.linspace(low_price, high_price, 9)

    def initalize_node(self, depth):
        return dict(
            datapoints=[], 
            depth=depth,
            left_child=None,
            right_child=None,
            price=None,
            children=False
        )

    def fit(self, X_train, prices, revenue_matrix=None):
        self.train_size = len(X_train)

        # discretize prices
        self.discretize_prices_(prices)

        # get revenue matrix for training data
        if self.pred_revenue_matrix is None:
            self.get_revenue_matrix(X_train)

        revenue_matrix = self.pred_revenue_matrix

        # get best price index
        best_price = np.argmax(revenue_matrix.sum(axis=0))
        # get best revenue
        best_R = np.max(revenue_matrix.sum(axis=0))

        # create root node
        self.root_node = dict(
            datapoints=np.arange(len(X_train)), 
            depth=0,
            left_child=None,
            right_child=None,
            price=best_price,
            revenue=best_R
        )

        def split(parent):
            """
            Split parent node into binary children
            """
            # initialize right and left children
            depth = parent['depth'] + 1
            left_child = self.initalize_node(depth)
            right_child = self.initalize_node(depth)

            # get parent node datapoints indices
            S = parent['datapoints']

            # get corresponding data
            X = X_train.to_numpy()[S, :]
            n_observations, n_features = X.shape

            # search every single feature 
            # and every single observation
            # for the optimal split
            best_split_revenue = 0
            for i in range(n_observations):
                for j in range(n_features):
                    # get children sets
                    S1 = S[np.flatnonzero(X[:, j] <= X[i, j])]
                    S2 = S[np.flatnonzero(X[:, j] > X[i, j])]

                    best_price1, best_R1 = self.get_best_price_and_revenue(X_train.iloc[S1, :])
                    best_price2, best_R2 = self.get_best_price_and_revenue(X_train.iloc[S2, :])

                    if (best_price1 is not None) & (best_price2 is not None):
                        # if set is empty associated revenue 
                        # is defined as 0
                        total_revenue = np.nan_to_num(best_R1, 0) + np.nan_to_num(best_R2, 0)

                        if self.is_risk_controlled(total_revenue, best_split_revenue):
                            # update best split
                            parent['split_value'] = X[i, j]
                            parent['split_feature'] = j

                            # raise exception if partitioning is 
                            # improperly performed
                            if set(S1.tolist() + S2.tolist()) != set(S):
                                raise Exception("Disallowed Split")

                            # update children datapoints
                            left_child['datapoints'] = S1
                            right_child['datapoints'] = S2

                            # update children best greedy price indices
                            left_child['price'] = best_price1
                            right_child['price'] = best_price2

                            # update best greedy revenue
                            left_child['revenue'] = best_R1
                            right_child['revenue'] = best_R2

                            # update optimal greedy split revenue
                            best_split_revenue = total_revenue
           
            a = len(left_child['datapoints']) > 0
            b = len(right_child['datapoints']) > 0
            if a & b:
                # if split was successful
                # update children indicator
                parent['children'] = True
                # if children depth is less than
                # max split both chilren again
                if depth < self.max_depth:
                    # update parent children
                    parent['left_child'] = left_child
                    parent['right_child'] = right_child

                    # recursively iterate
                    split(left_child)
                    split(right_child)

        # recursively split training data
        split(self.root_node)

    def get_test_nodes(self, X_test, max_depth):
        """
        Build a second tree using test data
        and first tree
        """
        # copy root node
        root_node1 = self.root_node.copy()
        root_node2 = dict(
            datapoints=np.arange(len(X_test)), 
            depth=0,
            left_child=None,
            right_child=None,
            revenue=None
        )

        revenue_matrix = self.get_revenue_matrix(X_test)

        # intitialize list to collect leave nodes
        leaf_nodes = []

        def split(parent1, parent2):
            """
            parent1: from train set
            parant2: from test set
            """
            # get children depth
            depth = parent1['depth']
            # get parent node datapoints indices
            S = parent2['datapoints']

            if (depth < max_depth) & (parent1['children']):
                    
                # get corresponding data
                X = X_test.drop('price', axis=1).to_numpy()[S, :]

                # split data
                j = parent1['split_feature']
                v = parent1['split_value']

                # get price assignment
                k1 = parent1['left_child']['price']
                k2 = parent1['right_child']['price']

                S1 = S[np.flatnonzero(X[:, j] <= v)]
                S2 = S[np.flatnonzero(X[:, j] > v)]

                # initialize right and left children
                left_child = dict(datapoints=[], depth=depth)
                right_child = dict(datapoints=[], depth=depth)

                # assign split datapoint
                left_child['datapoints'] = S1
                right_child['datapoints'] = S2

                # store best_price index
                left_child['price'] = k1
                right_child['price'] = k2

                # calculate total revenue
                left_child['revenue'] = revenue_matrix[S1, k1].sum()
                right_child['revenue'] = revenue_matrix[S2, k2].sum()

                # update parent children
                parent2['left_child'] = left_child
                parent2['right_child'] = right_child

                # recursively iterate
                split(parent1['left_child'], left_child)
                split(parent1['right_child'], right_child)

            else:
                leaf_nodes.append(parent2)  

        # initialize test root node price
        root_node2['price'] = root_node1['price']
        # call recursive split
        split(root_node1, root_node2)

        return leaf_nodes

    def predict_revenue(self, X_test):
        """
        Return average revenue per individual 
        from personalization
        """
        # get leave nodes
        leaf_nodes = self.predict(X_test)

        # get personalized pricing revenue
        rev = sum([leaf_node['revenue'] for leaf_node in leaf_nodes])   

        return rev / len(X_test)

    def get_optimal_revenue(self, X):
        """
        Return average revenue per individual 
        from optimal
        """
        # get revenue matrix for training data
        revenue_matrix = self.get_revenue_matrix(X)

        return revenue_matrix.max(axis=1).sum() / len(X)

    def get_nodes(self, depth=None):
        """
        depth: (int)
        Return a list of root nodes 
        corresponding to a depth.
        If depth is None then depth is 
        automatically set to max_depth
        """
        nodes = []
        if depth is None:
            depth = self.max_depth

        # start at root and go down
        # until depth is reach and
        # all corresponding nodes obtained
        def go_down(node):
            if node['depth'] == depth:
                nodes.append(node)
            else:
                if node['children']:
                    go_down(node['left_child'])
                    go_down(node['right_child'])
                else:
                    nodes.append(node)

        # recursively go down
        go_down(self.root_node)

        n_points = sum([len(node['datapoints']) for node in nodes])
        if n_points != self.train_size:
            warnings.warn('Found ' + str(n_points) + ', expecting ' + str(self.train_size))

        return nodes