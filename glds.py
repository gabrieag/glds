#!/usr/bin/env python

# Python implementation of the Gaussian mixture probability
# hypothesis density (GM-PHD) filter. This code is based on
# the technical report by Ghahramani and Hinton (1996).
#
# Z. Ghahramani and G. E. Hinton, "Parameter estimation for
# linear dynamical systems," Department of Computer Science,
# University of Toronto, CRG-TR-96-2, 1996.

import numpy

from numpy import linalg,random

class sys():

    class model():

        def __init__(self,gain,noise):

            self.gain=gain
            self.noise=noise

        def sim(self,pred):

            # Store the size of the model.
            numresp,numpred=numpy.size(self.gain)

            cholfact=linalg.cholesky(self.noise)

            # Simulate a response.
            resp=numpy.dot(self.gain,pred)+\
                numpy.dot(cholfact,random.randn(numpred))

            return resp

        def prop(self,mean,var):

            covar=var.dot(self.gain.transpose())

            # Propagate the mean and variance through the model.
            mean=self.gain.dot(mean)
            var=self.gain.dot(covar)+self.noise

            return mean,var,covar

        def cond(self,mean,var,resp,uncer=0.0):

            # Store the size of the model.
            numresp,numpred=numpy.size(self.gain)

            covar=numpy.zeros([numpred,numresp])

            # Store the indices to the observed responses.
            obs=numpy.logical_not(numpy.isnan(resp))

            if not obs.any():
                return 0.0,mean,var,covar

            ind=numpy.ix_(obs,obs)

            # Compute the statistics of the innovation.
            innovmean=numpy.dot(self.gain[obs,:],mean)
            kalmgain=numpy.dot(var,self.gain[obs,:].transpose())
            innovvar=numpy.dot(self.gain[obs,:],kalmgain)+self.noise[ind]

            cholfact=numpy.linalg.cholesky(innovvar)

            # Evaluate the negative log-likelihood of the observations.
            negloglik=(obs.sum()/2.0)*math.log(2.0*math.pi)+numpy.sum(numpy.log(cholfact.diagonal()))\
                +numpy.sum(numpy.abs(linalg.solve(cholfact,resp[obs]-innovmean))**2)/2.0

            # Construct the Kalman and Joseph gain matrices.
            kalmgain=linalg.solve(innovvar,kalmgain.transpose()).transpose()
            josgain=numpy.eye(numpred)-numpy.dot(kalmgain,self.gain[obs,:])

            # Condition the mean and variance on the observations.
            mean=numpy.dot(josgain,mean)+numpy.dot(kalmgain,resp[obs])
            var=numpy.dot(josgain,numpy.dot(var,josgain.transpose()))+\
                numpy.dot(kalmgain,numpy.dot(self.noise[ind],kalmgain.transpose()))

            if uncer!=0.0:

                # Compute the predictor/response co-variance.
                covar[:,obs]=kalmgain.dot(uncer[ind])

            return const,mean,var,covar

        def comp(self,mean,var,covar,resp):

            # Store the indices to the missing and observed responses.
            miss=numpy.isnan(resp)
            obs=numpy.logical_not(miss)

            if miss.all():
                return mean,var,covar

            # Store the size of the model.
            numresp,numpred=numpy.size(self.gain)

            kalmgain=numpy.eye(numresp)
            josgain=numpy.eye(numresp)

            # Fill in the Kalman and Joseph gain matrices.
            ind=numpy.ix_(miss,obs)
            kalmgain[ind]=linalg.solve(self.noise[numpy.ix_(obs,obs)],
                self.noise[ind].transpose()).transpose()
            josgain[:,obs]=josgain[:,obs]-kalmgain[:,obs]

            # Compute the predictor/response co-variance.
            covar=covar.dot(josgain.transpose())

            # Condition the response mean/variance on the observations.
            mean=josgain.dot(mean)+numpy.dot(kalmgain[:,obs],resp[obs])
            var=numpy.dot(josgain,numpy.dot(var,josgain.transpose()))

            return mean,var,covar

    def __init__(self,initmean,initvar,transgain,transnoise,measgain,measnoise):

        # Check the initial mean.
        try:
            numstate,=numpy.shape(initmean)
        except ValueError:
            raise Exception('Initial mean must be a vector.')

        # Check the initial variance.
        if numpy.shape(initvar)!=(numstate,numstate):
            raise Exception('Initial variance must be a {}-by-{} matrix.'.format(numstate,numstate))
        if not numpy.allclose(numpy.transpose(initvar),initvar):
            raise Exception('Initial variance matrix must be symmetric.')
        try:
            cholfact=linalg.cholesky(initvar)
        except linalg.LinAlgError:
            raise Exception('Initial variance matrix must be positive-definite.')

        # Check the transition gain.
        if numpy.ndim(transgain)!=2 or numpy.shape(transgain)!=(numstate,numstate):
            raise Exception('Transition gain must be a {}-by-{} matrix.'.format(numstate,numstate))

        # Check the transition noise.
        if numpy.ndim(transnoise)!=2 or numpy.shape(transnoise)!=(numstate,numstate):
            raise Exception('Transition noise must be a {}-by-{} matrix.'.format(numstate,numstate))
        if not numpy.allclose(numpy.transpose(transnoise),transnoise):
            raise Exception('Transition noise matrix must be symmetric.')
        if numpy.any(linalg.eigvalsh(transnoise)<0.0):
            raise Exception('Transition noise matrix must be positive-semi-definite.')

        # Check the measurement gain.
        try:
            numobs,numcol=numpy.shape(measgain)
        except ValueError:
            raise Exception('Measurement gain must be a matrix.')
        if numcol!=numstate:
            raise Exception('Measurement gain matrix must have {} columns.'.format(numstate))

        # Check the measurement noise.
        if numpy.ndim(measnoise)!=2 or numpy.shape(measnoise)!=(numobs,numobs):
            raise Exception('Measurement noise must be a {}-by-{} matrix.'.format(numobs,numobs))
        if not numpy.allclose(numpy.transpose(measnoise),measnoise):
            raise Exception('Measurement noise matrix must be symmetric.')
        try:
            cholfact=linalg.cholesky(measnoise)
        except linalg.LinAlgError:
            raise Exception('Measurement noise matrix must be positive-definite.')

        # Set the model.
        self.initmean=numpy.asarray(initmean)
        self.initvar=numpy.asarray(initvar)
        self.transgain=numpy.asarray(transgain)
        self.transnoise=numpy.asarray(transnoise)
        self.measgain=numpy.asarray(measgain)
        self.measnoise=numpy.asarray(measnoise)

        self.__size__=numstate,numobs

    def sim(self,numpoint):

        numstate,numobs=self.__size__

        # Create transition/measurement models.
        trans=sys.model(self.transgain,self.transnoise)
        meas=sys.model(self.measgain,self.measnoise)

        state=numpy.zeros([numstate,numpoint+1])
        obs=numpy.zeros([numobs,numpoint])

        state[:,0]=self.initmean+numpy.dot(linalg.cholesky(self.initvar),random.randn(numstate))

        # Generate a sequence of data.
        for i in range(numpoint):
            state[:,i+1]=trans.sim(state[:,i])
            obs[:,i]=meas.sim(state[:,i+1])

        return state,obs

    def filt(self,obs):

        numstate,numobs=self.__size__

        # Check the observations.
        try:
            numrow,numpoint=numpy.shape(obs)
        except ValueError:
            raise Exception('Observations must be a matrix.')
        if numrow!=numobs:
            raise Exception('Observations matrix must have {} rows.'.format(numobs))

        # Create transition/measurement models.
        trans=sys.model(self.transgain,self.transnoise)
        meas=sys.model(self.measgain,self.measnoise)

        negloglik=numpy.zeros(numpoint)

        mean=numpy.zeros([nstate,numpoint+1])
        var=numpy.zeros([numstate,numstate,numpoint+1])

        # Initialize the states.
        mean[:,0]=self.initmean
        var[:,:,0]=self.initvar

        # Execute the forward recursions.
        for i in range(nelem):
            mean[:,i+1],var[:,:,i+1],a=trans.prop(mean[:,i],var[:,:,i])
            negloglik[i],mean[:,i+1],var[:,:,i+1],b=\
                meas.cond(mean[:,i+1],var[:,:,i+1],obs[:,i])

        return negloglik,(mean,var)

    def smooth(self,obs):

        numstate,numobs=self.__size__

        # Check the observations.
        try:
            numrow,numpoint=numpy.shape(obs)
        except ValueError:
            raise Exception('Observations must be a matrix.')
        if numrow!=numobs:
            raise Exception('Observations matrix must have {} rows.'.format(numobs))

        # Create transition/measurement models.
        trans=sys.model(self.transgain,self.transnoise)
        meas=sys.model(self.measgain,self.measnoise)

        negloglik=numpy.zeros(numpoint)

        mean=numpy.zeros([numstate,numpoint+1])
        var=numpy.zeros([numstate,numstate,numpoint+1])
        covar=numpy.zeros([numstate,numstate,numpoint])

        # Initialize the states.
        mean[:,0]=self.initmean
        var[:,:,0]=self.initvar

        # Execute the forward recursions.
        for i in range(numpoint):
            mean[:,i+1],var[:,:,i+1],covar[:,:,i]=trans.prop(mean[:,i],var[:,:,i])
            negloglik[i],mean[:,i+1],var[:,:,i+1],a=\
                meas.cond(mean[:,i+1],var[:,:,i+1],obs[:,i])

        # Execute the backward recursions.
        for i in range(numpoint,0,-1):
            # mu,sigma,omega=meas.prop(mean[:,i],var[:,:,i])
            # mu,sigma,omega=meas.comp(mu,sigma,omega,obs[:,i-1])
            b,mean[:,i-1],var[:,:,i-1],covar[:,:,i-1]=\
                trans.cond(mean[:,i-1],var[:,:,i-1],mean[:,i],var[:,:,i])

        return const,(mean,var,covar)

