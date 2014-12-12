
import numpy
import scipy.optimize
from lds import lds

if __name__=="__main__":

    dim=2

    """ Set size. """
    nseq=10
    nelem=50

    """ Set options. """
    step=0.5
    reg=1e-2

    def model(param):

        """ Recover parameters. """
        trans,obs=param

        """ Instantiate linear dynamical system. """
        obj=lds(2*dim,dim)

        gain=numpy.array([(1.0,step),(0.0,1.0)])
        noise=(step*trans)**2*numpy.array([(step**2/4.0,step/2.0),
                                           (step/2.0,1.0+reg**2)])

        """ Set transition model. """
        obj.trans.gain=numpy.kron(gain,numpy.eye(dim))
        obj.trans.noise=numpy.kron(noise,numpy.eye(dim))

        """ Set observation model. """
        obj.obs.gain=numpy.kron(numpy.array([1.0,0.0]),numpy.eye(dim))
        obj.obs.noise=numpy.kron(numpy.array([obs**2]),numpy.eye(dim))

        return obj

    param=[0.5,1.0]

    """ Build model for sampling. """
    obj=model(param)

    print('True parameters: '.rjust(25),*param)

    seq=[]

    """ Generate data. """
    for i in range(nseq):
        state,feat=obj.sim(nelem)
        seq.append(feat)

    def cost(param):

        """ Build model with given parameters. """
        obj=model(param)

        val=0.0

        """ Evaluate cost. """
        for feat in seq:
            const,mean,var=obj.filt(feat)
            val+=const.sum()

        return val

    """ Fit parameters. """
    param=[2.0,5.0]
    result=scipy.optimize.minimize(cost,param,method='Nelder-Mead')

    if result.success:

        print('Estimated parameters: '.rjust(25),*result.x)
