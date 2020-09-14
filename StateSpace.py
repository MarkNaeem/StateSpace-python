import numpy as np
import matplotlib.pyplot as plt

class StateSpace():
    def __init__(self,A,B,C,D,dt,x0=None):        
        """The class initializer.
          Parameters:
          -----------
          A,B,C,D: numpy array, state-space system matrices. 
          dt:      int, the fixed time step of the system.
          x0:      numpy array, initial state, if None zero initial state will be used."""

        #zero initial state if it wasn't given
        if x0==None: x0 = np.zeros((A.shape[0],1))

        #state space matrices
        assert A.shape[0]==A.shape[1] , "A must be square matrix"
        assert x0.shape  ==(A.shape[0],1), "Initial state must have shape " + str([A.shape[0],1])
        assert A.shape[0]==B.shape[0] , "A and B must have same rows number"
        assert C.shape[0]==D.shape[0] , "C and D must have same rows number"
        assert C.shape[1]==A.shape[1] , "A and C must have same coulmns number"
        assert B.shape[1]==D.shape[1] , "A and C must have same coulmns number"

        self.A = A 
        self.B = B 
        self.C = C
        self.D = D
        self.dt = dt
        #to ease calculations in every iteration
        self.__inv__ = np.linalg.inv(np.eye(A.shape[0])-A*dt)
        #loggers
        self.x=[x0]
        self.y=[]
        self.u=[]
        
    def reset(self,x0):        
        """resets everything in the system, clears all input, output, and state list.
          x0:      numpy array, initial state."""
        #loggers
        self.x=[x0]
        self.y=[]
        self.u=[]

    def step(self,u):
        """solves the system for a single time step given a single time step input.
           parameters:
           -----------
           u: numpy array, single step input, must be in shape (B.shape[0],) or (B.shape[0],1)."""
        self.u.append(u)
        self.x.append( np.dot(self.__inv__ , (np.dot( self.B*self.dt,u)+self.x[-1]) ) )
        self.y.append(self.C.dot(self.x[-1])+self.D.dot(u))
 
    def solve(self,U):
        """solves the system for a given input.
           parameters:
           -----------
           u: list or numpy array, input values, must be in shape (#steps,B.shape[0]) or a list of single step inputs in shape 
              (B.shape[0],) or (B.shape[0],1). """
        for u in U: self.step(u)

    def plot(self, input_labels=None, output_labels=None, plot_state=True):
        """plots the input given to the system so far, optionally plots the system state, and the system output.
           state labels are names as 'state1', 'state2'...and so on, so as input and output labels if left to None.
           parameters:
           ----------- 
           input_labels:  list, labels to be printed on the input plots.
           output_labels: list, labels to be printed on the output plots.
           plot_state:    bool, whether to plot system state or not.
           """

        if output_labels==None :       output_labels = ['Output'+str(i) for i in range(self.C.shape[0])] 
        elif type(output_labels)==str: list(output_labels)

        if input_labels==None : input_labels  = ['input'+str(i) for i in range(self.B.shape[1])]  
        elif type(input_labels)==str: list(input_labels)
 
        Time    = np.arange(0,len(self.y)*dt,dt)
        inputs  = np.array(self.u).T.reshape(-1,len(self.u))
        outputs = np.array(self.y).T.reshape(-1,len(self.y))

        for inp,label in zip(inputs,input_labels):          
            plt.plot(Time,inp ,'b', label =  label )
            plt.xlabel("time (s)")
            plt.ylabel(label)
            plt.grid(1)
            plt.show()

        for op,label in zip(outputs,output_labels):          
            plt.plot(Time,inp ,'r', label =  label )
            plt.xlabel("time (s)")
            plt.ylabel(label)
            plt.grid(1)
            plt.show()
        
        if plot_state:
           state_labels  = ['state'+str(i) for i in range(self.A.shape[0])]  
           state = np.array(self.x).T.reshape(-1,len(self.x)) 
           for st,label in zip(state,state_labels):          
               plt.plot(Time,inp ,'g', label =  label )
               plt.xlabel("time (s)")
               plt.ylabel(label)
               plt.grid(1)
               plt.show()

    def get_output(self):
      """returns the system output."""
      return np.array(self.y).reshape(-1,self.C.shape[0])


    def get_state(self):
      """returns the system state."""
      return np.array(self.x)

    def get_input(self):
      """returns the system input given so far."""
      return np.array(self.u)
