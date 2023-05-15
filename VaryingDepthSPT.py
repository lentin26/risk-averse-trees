
from SyntheticData import SyntheticData
from EvaluateSPT import EvaluateSPT
import lightgbm as lgb
from tqdm import tqdm


class VaryingDepthSPT:

    def __init__(self):
        self.data_generator = SyntheticData()

    def get_varying_depth_rev(self, model_id, n_samples, n_iter, depths=None):
        spt_results = []
        opt_results = []

        # take multiple samples
        for _ in tqdm(range(n_iter)):
            # generate dataset
            data = self.data_generator.generate_data(model_id=model_id, n_samples=n_samples)

            # get covariates, target and price thresholds
            covariates = [x for x in data.columns if x not in ['optimal_price', 'Y']]
            # print("Covariates:", covariates)
            X = data[covariates]
            y = data['Y']
            price_thresh = data['optimal_price']

            # lightGBM
            num_round = 50 
            model = lgb.LGBMClassifier(n_estimators=num_round)

            # fix imbalance over training set only
            model.fit(X, y)

            # evaluate model at various depths
            if depths is None:
                depths = [1, 2, 3, 4, 5]

            # instantiate evaluator
            eval = EvaluateSPT(X, y, price_thresh)
            # build predictive and ground truth (optimal) trees
            eval.fit_trees(model, max_depth=max(depths))

            spt_revenues = []
            opt_revenues = []
            for depth in depths:
                # get results
                spt_revenue, opt_revenue = eval.evaluateSPT(depth=depth)
                # append results
                spt_revenues.append(spt_revenue)
                opt_revenues.append(opt_revenue)

            # append final results for sample
            spt_results.append(spt_revenues)
            opt_results.append(opt_revenues)

        return spt_results, opt_results