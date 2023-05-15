import numpy as np
import warnings


class StudentPrescriptionTree():
    def __init__(self, teacher_model, max_depth):
        self.teacher_model = teacher_model
        self.max_depth = max_depth
        self.root_node = None
        self.prices = None
        self.train_size = None

    def get_revenue_matrix(self, X):
        # get descrete prices
        prices = self.prices

        # make empty response matrix
        revenue_matrix = np.empty((len(X), len(prices)))

        for i in range(len(prices)):
            counterfactual = X.copy()
            price = prices[i]
            counterfactual.price = price

            # construct reponse matrix
            revenue_matrix[:, i] = price * self.teacher_model.predict_proba(counterfactual)[:, 1]

        return revenue_matrix

    def discretize_prices_(self, X_train):
        # discretize prices
        low_price = X_train.price.quantile(0.10)
        high_price = X_train.price.quantile(0.90)

        self.prices = np.linspace(low_price, high_price, 9)

    def fit(self, X_train, revenue_matrix=None):
        self.train_size = len(X_train)

        # discretize prices
        self.discretize_prices_(X_train)

        # get revenue matrix for training data
        if revenue_matrix is None:
            revenue_matrix = self.get_revenue_matrix(X_train)

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
            left_child = dict(
                datapoints=[], 
                depth=depth,
                left_child=None,
                right_child=None,
                children=False)
            right_child = dict(
                datapoints=[], 
                depth=depth,
                left_child=None,
                right_child=None,
                children=False)

            # get parent node datapoints indices
            S = parent['datapoints']

            # get corresponding data
            X = X_train.drop('price', axis=1).to_numpy()[S, :]
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
                    # now that we've split the data
                    # search for the best price-combo using
                    # pre-calculated revenue matrix
                    # get revenue at each price point
                    R1 = revenue_matrix[S1, :].sum(axis=0)
                    R2 = revenue_matrix[S2, :].sum(axis=0)

                    # get best price index
                    best_price1 = np.argmax(R1)
                    best_price2 = np.argmax(R2)
                    # get best revenue
                    best_R1 = np.max(R1)
                    best_R2 = np.max(R2)

                    # if set is empty associated revenue 
                    # is defined as 0
                    total_revenue = np.nan_to_num(best_R1, 0) + np.nan_to_num(best_R2, 0)
    
                    if total_revenue > best_split_revenue:
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

            # check that revenue has not decreased
            if parent['revenue'] - best_split_revenue > 0.001:
                raise Exception("Disallowed revenue decrease.",
                                total_revenue, parent['revenue'])

            if depth < self.max_depth:
                a = len(left_child['datapoints']) > 0
                b = len(right_child['datapoints']) > 0
                if a & b:
                    # update parent children
                    parent['left_child'] = left_child
                    parent['right_child'] = right_child

                    # update children indicator
                    parent['children'] = True

                    # recursively iterate
                    split(left_child)
                    split(right_child)

        # recursively split training data
        split(self.root_node)

    def predict(self, X_test):
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

        # get revenue matrix for test data
        revenue_matrix = self.get_revenue_matrix(X_test)

        # intitialize list to collect leave nodes
        leaf_nodes = []

        def split(parent1, parent2):
            """
            parent1: from train set
            parant2: from test set
            """
            # get children depth
            depth = parent1['depth'] + 1
            # get parent node datapoints indices
            S = parent2['datapoints']
            #if depth - 1 < self.max_depth:
            if parent1['children']:
                
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