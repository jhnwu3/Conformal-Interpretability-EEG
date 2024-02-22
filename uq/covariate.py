from uq.conformal import *
from abc import ABC, abstractmethod
from interpret.chefer import *
# should return B x 1 likelihoods ratio of being from test or calibration set
def likelihood_ratio(input, kde_cal, kde_test) -> np.ndarray:
    return (kde_test(input) / kde_cal(input))

def sum_likelihood_ratios(cal_dataloader, kde_cal, kde_test):
    sum = []
    # sum them all up
    for batch, label in cal_dataloader:
        sum.append(likelihood_ratio(batch, kde_cal, kde_test))
    return np.sum(np.concatenate(sum))

def calibration_weight(input, sum_cal_likelihoods ,kde_cal, kde_test) -> torch.Tensor:
    cal_wt = likelihood_ratio(input, kde_cal, kde_test) / sum_cal_likelihoods
    return torch.from_numpy(cal_wt) # must be tensor to multiply with output of calibration score

def test_weight(input, sum_cal_likelihoods, kde_cal, kde_test) -> torch.Tensor:
    test_wt = likelihood_ratio(input, kde_cal, kde_test)
    test_wt /= (sum_cal_likelihoods + likelihood_ratio(input, kde_cal, kde_test))
    return torch.from_numpy(test_wt)

class Covariate(ABC):
    def __init__(self, interpreter : STTransformerInterpreter, cal_density_estimator, test_density_estimator):
        self.interpreter = interpreter
    @abstractmethod
    def cal_score(self, input, class_index):
        pass
    @abstractmethod
    def score(self, input):
        pass

class SoftMaxCovariate(Covariate):
    def __init__(self, interpreter : STTransformerInterpreter, 
                 cal_density_estimator, test_density_estimator, dataloader):
        
        self.interpreter = interpreter
        self.cal_density_estimator = cal_density_estimator
        self.test_density_estimator = test_density_estimator
        self.softmax = SoftMax(interpreter)
        self.sum_cal_likelihoods = sum_likelihood_ratios(dataloader, self.cal_density_estimator, self.test_density_estimator)

    def cal_score(self, input, class_index):
        cal_score = self.softmax.cal_score(input, class_index)
        return calibration_weight(input, self.sum_cal_likelihoods, self.cal_density_estimator, self.test_density_estimator) * cal_score
    # every input should in principle 
    def score(self, input):
        score = self.softmax.score(input)
        return test_weight(input, self.sum_cal_likelihoods, self.cal_density_estimator, self.test_density_estimator).unsqueeze(1) * score


class CumulativeSoftMaxCovariate(Covariate):

    def __init__(self, interpreter : STTransformerInterpreter, cal_density_estimator, test_density_estimator, dataloader):
        self.interpreter = interpreter
        self.cal_density_estimator = cal_density_estimator
        self.test_density_estimator = test_density_estimator
        self.cum_softmax = CumulativeSoftMax(interpreter)
        self.sum_cal_likelihoods = sum_likelihood_ratios(dataloader, self.cal_density_estimator, self.test_density_estimator)

    def cal_score(self, input, class_index):
        cal_score = self.cum_softmax.cal_score(input, class_index)
        return calibration_weight(input, self.sum_cal_likelihoods, self.cal_density_estimator, self.test_density_estimator) * cal_score
    
    def score(self, input):
        score = self.cum_softmax.score(input)
        return test_weight(input, self.sum_cal_likelihoods, self.cal_density_estimator, self.test_density_estimator).unsqueeze(1) * score

