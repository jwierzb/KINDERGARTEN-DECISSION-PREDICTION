# Wrapped C50 class from R imported with rpy2 package
import rpy2.robjects as robjects
import csv

# Import R C5.0 functions
# import rpy2's package module
import rpy2.robjects.packages as rpackages


# import R's utility package
utils = rpackages.importr('utils')

# select a mirror for R packages
utils.chooseCRANmirror(ind=1) # select the first mirror in the list

# R package names
packnames = ('c50', 'mvtnorm', 'grDevices')


# R vector of strings
from rpy2.robjects.vectors import StrVector
from rpy2.robjects import r, pandas2ri


# Selectively install what needs to be install.
# We are fancy, just because we can.
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))


C50 = rpackages.importr('C50')
clf = robjects.r('C5.0')
predictor = robjects.r('predict')
summary = robjects.r('summary')
controlClf = robjects.r('C5.0Control')
plotter = robjects.r('plot')
png = robjects.r('png')
devOff = robjects.r('dev.off')
as_character = robjects.r('as.character')
pandas2ri.activate()


class C50:
    def __init__(self, X, Y): # X is dataframe with samples, Y with labels
        self.X = X
        self.Y = Y
        self.tree = None
        self.enabled = False

    def train(self, trials=1, subset=1):
        if subset == 1:
            self.tree = clf(pandas2ri.py2ri(self.X), robjects.FactorVector(self.Y.values), trials=trials)
        elif 0 < subset < 1:
            self.tree = clf(pandas2ri.py2ri(self.X), robjects.FactorVector(self.Y.values), trials=trials,
                            control=controlClf(subset=subset))
        else:
            raise ValueError('Bad value error')
        self.enabled = True

    def predict(self, x): # takes dataframe of rows to predict, returns
        return [as_character(predictor(self.tree, pandas2ri.py2ri(x.loc[i:i])))[0] for i in range(0, len(x))]

    def plot_tree(self, file_name = "default.png"):
        png(file_name, width=3000, height=3000)
        plotter(self.tree)
        devOff()

    def print_summary(self, file_name="summary.txt"):
        with open(file_name, mode='w+') as f:
            print(summary(self.tree), file=f)
        # print(summary(self.tree))

    def predict_and_save(self, x, file_name="result.txt"):
        result = self.predict(x)
        with open(file_name, mode='w') as f:
            for i in result:
                f.write(i+'\n')

    def predict_to_csv(self, X, file_name="result.csv"):
        result = self.predict(X)
        with open(file_name, mode='w') as f:
            f = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
            for i in range(0, len(result)):
                tab = [p for p in X.values[i]]
                tab.append(result[i])
                f.writerow(tab)
