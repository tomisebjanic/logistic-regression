import numpy
from scipy.optimize import fmin_l_bfgs_b
import pylab


def draw_decision(X, y, classifier, at1, at2, grid=50):
    points = numpy.take(X, [at1, at2], axis=1)
    maxx, maxy = numpy.max(points, axis=0)
    minx, miny = numpy.min(points, axis=0)
    difx = maxx - minx
    dify = maxy - miny
    maxx += 0.02*difx
    minx -= 0.02*difx
    maxy += 0.02*dify
    miny -= 0.02*dify

    for c,(x,y) in zip(y,points):
        pylab.text(x,y,str(c), ha="center", va="center")
        pylab.scatter([x],[y],c=["b","r"][c!=0], s=200)

    num = grid
    prob = numpy.zeros([num, num])
    for xi,x in enumerate(numpy.linspace(minx, maxx, num=num)):
        for yi,y in enumerate(numpy.linspace(miny, maxy, num=num)):
            #probability of the closest example
            diff = points - numpy.array([x,y])
            dists = (diff[:,0]**2 + diff[:,1]**2)**0.5 #euclidean
            ind = numpy.argsort(dists)
            prob[yi,xi] = classifier(X[ind[0]])[1]

    pylab.imshow(prob, extent=(minx,maxx,maxy,miny))

    pylab.xlim(minx, maxx)
    pylab.ylim(miny, maxy)
    pylab.xlabel(at1)
    pylab.ylabel(at2)

    pylab.show()


def load(name):
    """
    Open the file. Return a data matrix X (columns are features)
    and a vector of classes.
    """
    data = numpy.loadtxt(name)
    X, y = data[:,:-1], data[:,-1].astype(numpy.int)
    return X,y


def h(x, theta):
    """
    Predict the probability for class 1 for the current instance
    and a vector theta.
    """
    # return 1 / (1 + numpy.e ** -sum([ti * xi for xi, ti in zip(x, theta)]))
    return 1/ (1 + numpy.exp(-x.transpose().dot(theta)))


def cost(theta, X, y, lambda_):
    """
    Return the value of the cost function. Because the optimization algorithm
    used can only do minimization you will have to slightly adapt equations from
    the lectures.
    """
    # return (-sum([yi*numpy.log(h(xi, theta)) + (1-yi)*numpy.log(1-h(xi, theta)) for xi, yi in zip(X, y)])
    #         + (lambda_/2*X.shape[0]) * sum([ti**2 for ti in theta]))/ X.shape[0]
    p = h(X.transpose(), theta)
    reg = 4.0*lambda_*theta.dot(theta) # [ti**2 for ti in theta]
    return (-numpy.log(numpy.where(y, p, 1-p)).sum() + reg)/X.shape[0]


def grad(theta, X, y, lambda_):
    """
    The gradient of the cost function. Return a numpy vector of the same
    size at theta.
    """
    # Xt = X.transpose()
    # j = numpy.array([(-sum((y - h(Xt, theta))*Xt[i]) )/X.shape[0] for i in range(len(theta))])
    # regularization = numpy.array([2*ti for ti in theta]) * (4*lambda_/X.shape[0])
    # regularization = (theta*2.0) * (4.0*lambda_/X.shape[0])
    # return j + regularization
    Xt = X.transpose()
    p = h(Xt, theta)
    reg = (theta*2.0) * (4.0*lambda_/X.shape[0]) # [2*ti for ti in theta]
    return -Xt.dot(y-p)/X.shape[0] + reg


def CA(real, predictions):
    """
    CA = (TP + TN) / m
    """
    tp, tn = 0, 0
    for i in range(len(predictions)):
        if (predictions[i][0] > predictions[i][1]) and (predictions[i][0] >= 0.5 and real[i] == 0):
            tn += 1
        elif (predictions[i][0] < predictions[i][1]) and (predictions[i][1] >= 0.5 and real[i] == 1):
            tp += 1

    return (tp + tn) / len(real)


def test_cv(learner, X, y, k=5):
    n = X.shape[0] / k
    ix = 0
    results = []
    for i in range(k):
        Xtest = X[ix:ix+n]
        Xtrain = numpy.delete(X, list(range(int(ix), int(ix+n))), axis=0)
        ytrain = numpy.delete(y, list(range(int(ix), int(ix+n))))

        c = learner(Xtrain, ytrain)
        for x in Xtest:
            results.append(c(x))
        ix += n
    return results


def test_learning(learner, X, y):
    c = learner(X,y)
    results = [ c(x) for x in X ]
    return results


def part2():
    X,y = load('reg.data')
    learner = LogRegLearner(lambda_=0.)
    classifier = learner(X,y)
    draw_decision(X, y, classifier, 0, 1)


def part4():
    X,y = load('GDS1059.data')
    lambdas = [0, 0.0005, 0.005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.05, 0.01, 0.02, 0.03,
               0.04, 0.06, 0.07, 0.08, 0.09, 0.1, 0.5, 1., 5., 10., 50., 100., 1000.]
    best_lambda, best_ca = -1, -1

    for l in lambdas:
        learner = LogRegLearner(lambda_=l)
        a = test_cv(learner, X, y)
        ca = CA(y, a)
        if ca > best_ca:
            best_ca = ca
            best_lambda = l

    print('Best CA:', best_ca)
    print('Best lambda:', best_lambda)
    learner = LogRegLearner(lambda_=best_lambda)
    classifier = learner(X, y)
    draw_decision(X, y, classifier, 0, 20)


class LogRegClassifier(object):

    def __init__(self, th):
        self.th = th

    def __call__(self, x):
        """
        Predict the class for a vector of feature values.
        Return a list of [ class_0_probability, class_1_probability ].
        """
        x = numpy.hstack(([1.], x))
        p1 = h(x, self.th)
        return [ 1-p1, p1 ]


class LogRegLearner(object):

    def __init__(self, lambda_=0.0):
        self.lambda_ = lambda_

    def __call__(self, X, y):
        """
        Build a prediction model for date X with classes y.
        """
        X = numpy.hstack((numpy.ones((len(X),1)), X))

        #optimization as minimization
        theta = fmin_l_bfgs_b(cost,
            x0=numpy.zeros(X.shape[1]),
            args=(X, y, self.lambda_),
            fprime=grad)[0]

        return LogRegClassifier(theta)


if __name__ == "__main__":
    #
    # Usage example
    #

    # X,y = load('reg.data')
    # X, y = load('GDS1059.data')
    # learner = LogRegLearner(lambda_=0.0)
    # classifier = learner(X,y) # we get a model
    #
    # prediction = classifier(X[0]) # prediction for the first training example
    # print(prediction)
    part2()
    part4()
