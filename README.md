# Master-Thesis

This repository contains all the python files used for my master thesis.


To run the different libraries, the following python versions were used: 

| AutoML Framework | Python Version |   
|------------------|----------------|
| AutoSklearn      | 3.7            |
| TPOT             | 3.7            | 
| DSWIZARD         | 3.8            |  
| AlphaD3M         |                |  

I used a separate venv for each library. The corresponding dependencies are to be found under /config.

----------------------------
## Datasets 

Datasets can be downloaded by running 

```
python utils/download_openml_datasets.py
```
the directory `datasets` should be then automatically created for saving each dataset. 


------------------------------
## Running AutoML Frameworks

In each directory starting with `_`, run can be directly executed and the results should be saved automatically. 

note: before running DSWIZARD, make sure to run `./_dswizard/get_meta_learning_base.sh`


-------------------------- 

## Configs

All config files for AutoML framework, seed, dependencies are in `config`

-------------------------

## Changes to AutoML frameworks source code

I changed some of the sources code of frameworks to get more information out as follows:

1- Auto-Sklearn

This is changed to acquire the run statistics as a dict instead of a string 

``/home/hadi/PycharmProjects/Master-Thesis/venv/autosklearn_37/lib/python3.7/site-packages/autosklearn/automl.py``

```python
 def sprint_statistics(self) -> str:
        stats = {}
        check_is_fitted(self)
        cv_results = self.cv_results_
        sio = io.StringIO()
        sio.write("auto-sklearn results:\n")
        sio.write("  Dataset name: %s\n" % self._dataset_name)
        if len(self._metrics) == 1:
            sio.write("  Metric: %s\n" % self._metrics[0])
            stats["metric"] = [self._metrics[0]]
        else:
            sio.write("  Metrics: %s\n" % self._metrics)
            stats["metric"] = [self._metrics]

        idx_success = np.where(
            np.array(
                [
                    status
                    in ["Success", "Success (but do not advance to higher budget)"]
                    for status in cv_results["status"]
                ]
            )
        )[0]
        if len(idx_success) > 0:
            key = (
                "mean_test_score"
                if len(self._metrics) == 1
                else f"mean_test_" f"{self._metrics[0].name}"
            )

            if not self._metrics[0]._optimum:
                idx_best_run = np.argmin(cv_results[key][idx_success])
            else:
                idx_best_run = np.argmax(cv_results[key][idx_success])
            best_score = cv_results[key][idx_success][idx_best_run]
            sio.write("  Best validation score: %f\n" % best_score)

        stats["best_valid_score"] = [best_score]
        num_runs = len(cv_results["status"])
        sio.write("  Number of target algorithm runs: %d\n" % num_runs)
        stats["num_runs"] = [num_runs]

        num_success = sum(
            [
                s in ["Success", "Success (but do not advance to higher budget)"]
                for s in cv_results["status"]
            ]
        )
        sio.write("  Number of successful target algorithm runs: %d\n" % num_success)
        stats["num_success"] = [num_success]

        num_crash = sum([s == "Crash" for s in cv_results["status"]])
        sio.write("  Number of crashed target algorithm runs: %d\n" % num_crash)
        stats["num_crash"] = [num_crash]

        num_timeout = sum([s == "Timeout" for s in cv_results["status"]])
        sio.write(
            "  Number of target algorithms that exceeded the time "
            "limit: %d\n" % num_timeout
        )
        stats["num_timeout"] = [num_timeout]

        num_memout = sum([s == "Memout" for s in cv_results["status"]])
        sio.write(
            "  Number of target algorithms that exceeded the memory "
            "limit: %d\n" % num_memout
        )
        stats["num_memout"] = [num_memout]

        return sio.getvalue(), stats
```

2 - TPOT
This is changed to save each population in as .pkl

```/home/hadi/PycharmProjects/Master-Thesis/venv/38_tpot/lib/python3.7/site-packages/tpot/gp_deap.py```

```python
def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, pbar,
                   stats=None, halloffame=None, verbose=0,
                   per_generation_function=None, log_file=None):


    def pickle_pop(pop, ix):
        from pickle import dumps
        from os.path import join as pjoin

        gp_pop = list(map(lambda inv: [gp.graph(inv), inv.fitness.values[-1]], pop))

        out_dir = log_file.name.split("/")[:-1]
        out_dir = "/".join(out_dir)

        out_full_path = pjoin(out_dir, f"pops/{ix}.pkl")

        with open(out_full_path, "wb") as out_name:
            pickle.dump(gp_pop, out_name)



    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Initialize statistics dict for the individuals in the population, to keep track of mutation/crossover operations and predecessor relations
    for ind in population:
        initialize_stats_dict(ind)

    population[:] = toolbox.evaluate(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(population), **record)

    pickle_pop(population, 0)
    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the population
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)


        # Update generation statistic for all individuals which have invalid 'generation' stats
        # This hold for individuals that have been altered in the varOr function
        for ind in offspring:
            if ind.statistics['generation'] == 'INVALID':
                ind.statistics['generation'] = gen

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        offspring = toolbox.evaluate(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # pbar process
        if not pbar.disable:
            # Print only the best individual fitness
            if verbose == 2:
                high_score = max(halloffame.keys[x].wvalues[1] \
                    for x in range(len(halloffame.keys)))
                pbar.write('\nGeneration {0} - Current '
                            'best internal CV score: {1}'.format(gen,
                                                        high_score),

                            file=log_file)

            # Print the entire Pareto front
            elif verbose == 3:
                pbar.write('\nGeneration {} - '
                            'Current Pareto front scores:'.format(gen),
                            file=log_file)
                for pipeline, pipeline_scores in zip(halloffame.items, reversed(halloffame.keys)):
                    pbar.write('\n{}\t{}\t{}'.format(
                            int(pipeline_scores.wvalues[0]),
                            pipeline_scores.wvalues[1],
                            pipeline
                        ),
                        file=log_file
                    )

        # after each population save a periodic pipeline
        if per_generation_function is not None:
            per_generation_function(gen)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        pickle_pop(population, gen)
    return population, logbook


```

3- DSWIZARD
metric was changed from 'wminkowski' to 'minkowski' because it didnt work
```/home/hadi/PycharmProjects/Master-Thesis/venv/38_dswizard/lib/python3.8/site-packages/dswizard/core/similaritystore.py```

```python
class SimilarityStore:
    N_MF = 44

    def __init__(self, model: Optional[Pipeline]):
        self.mfs = None
        if model is not None:
            self.model: Pipeline = model
            # Remove OHE encoded algorithm
            self.weight = self.model.steps[-1][1].feature_importances_[0:SimilarityStore.N_MF]
            self.weight = (self.weight / self.weight.sum()) * SimilarityStore.N_MF
        else:
            self.model = None
            self.weight = np.ones(SimilarityStore.N_MF)
        self.neighbours = NearestNeighbors(metric='minkowski', p=2, metric_params={'w': self.weight})
        self.data = []

    def add(self, mf: MetaFeatures, data=None):
        mf_normal = self._normalize(mf)
        if self.mfs is None:
            self.mfs = mf_normal.reshape(1, -1)
        else:
            self.mfs = np.append(self.mfs, mf_normal, axis=0)
        self.neighbours.fit(self.mfs)
        self.data.append(data)

    def get_similar(self, mf: MetaFeatures):
        X = self._normalize(mf)

        distance, idx = self.neighbours.kneighbors(X, n_neighbors=1)
        return distance[0][0], idx[0][0], self.data[idx[0][0]]

    def _normalize(self, X):
        # remove unused MF
        Xt = X[:, 0:SimilarityStore.N_MF]
        if self.model is not None:
            for name, transform in self.model.steps[:2]:
                Xt = transform.transform(Xt)
        return Xt

```