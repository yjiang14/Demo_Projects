""""""  		  	   		     		  		  		    	 		 		   		 		  
"""  		  	   		     		  		  		    	 		 		   		 		  
Template for implementing QLearner  (c) 2015 Tucker Balch  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		     		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		     		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		     		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		     		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		     		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		     		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		     		  		  		    	 		 		   		 		  
or edited.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		     		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		     		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		     		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
Student Name: Yan Jiang (replace with your name)  		  	   		     		  		  		    	 		 		   		 		  
GT User ID: yjiang347 (replace with your User ID)  		  	   		     		  		  		    	 		 		   		 		  
GT ID: 903250461 (replace with your GT ID)  		  	   		     		  		  		    	 		 		   		 		  
"""  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
import random as rand  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
import numpy as np  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
class QLearner(object):  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    This is a Q learner object.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    :param num_states: The number of states to consider.  		  	   		     		  		  		    	 		 		   		 		  
    :type num_states: int  		  	   		     		  		  		    	 		 		   		 		  
    :param num_actions: The number of actions available..  		  	   		     		  		  		    	 		 		   		 		  
    :type num_actions: int  		  	   		     		  		  		    	 		 		   		 		  
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.  		  	   		     		  		  		    	 		 		   		 		  
    :type alpha: float  		  	   		     		  		  		    	 		 		   		 		  
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.  		  	   		     		  		  		    	 		 		   		 		  
    :type gamma: float  		  	   		     		  		  		    	 		 		   		 		  
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.  		  	   		     		  		  		    	 		 		   		 		  
    :type rar: float  		  	   		     		  		  		    	 		 		   		 		  
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.  		  	   		     		  		  		    	 		 		   		 		  
    :type radr: float  		  	   		     		  		  		    	 		 		   		 		  
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.  		  	   		     		  		  		    	 		 		   		 		  
    :type dyna: int  		  	   		     		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		     		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    def __init__(  		  	   		     		  		  		    	 		 		   		 		  
        self,  		  	   		     		  		  		    	 		 		   		 		  
        num_states=100,  		  	   		     		  		  		    	 		 		   		 		  
        num_actions=4,  		  	   		     		  		  		    	 		 		   		 		  
        alpha=0.2,  		  	   		     		  		  		    	 		 		   		 		  
        gamma=0.9,  		  	   		     		  		  		    	 		 		   		 		  
        rar=0.5,  		  	   		     		  		  		    	 		 		   		 		  
        radr=0.99,  		  	   		     		  		  		    	 		 		   		 		  
        dyna=0,  		  	   		     		  		  		    	 		 		   		 		  
        verbose=False,
        # verbose = True  		  	   		     		  		  		    	 		 		   		 		  
    ):  		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
        Constructor method  		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
        self.verbose = verbose  		  	   		     		  		  		    	 		 		   		 		  
        self.num_actions = num_actions  		  	   		     		  		  		    	 		 		   		 		  
        self.s = 0  		  	   		     		  		  		    	 		 		   		 		  
        self.a = 0  
        self.num_states = num_states
        

        #initial Q table
        self.Q = np.zeros((self.num_states,self.num_actions))	
        #initial Tc table
        # self.TC = np.reshape([0.00001]*self.num_states*self.num_actions*self.num_states, (self.num_states, self.num_actions, self.num_states))
        self.TC = np.zeros((self.num_states, self.num_actions, self.num_states))
        #initial T table
        self.T = np.zeros((self.num_states, self.num_actions, self.num_states))
        #initial R table
        self.R = np.zeros((self.num_states, self.num_actions))

        # transfer variables
        self.rar = rar
        self.radr = radr
        self.alpha = alpha
        self.gamma = gamma
        self.dyna = dyna
  		  	   		     		  		  		    	 		 		   		 		  
    def querysetstate(self, s):  		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
        Update the state without updating the Q-table  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
        :param s: The new state  		  	   		     		  		  		    	 		 		   		 		  
        :type s: int  		  	   		     		  		  		    	 		 		   		 		  
        :return: The selected action  		  	   		     		  		  		    	 		 		   		 		  
        :rtype: int  		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
        self.s = s

        #decide if we ignore the action and randomly choose one

        # if rand.uniform(0.0,1.0) <= self.rar:
        #     self.a = rand.randint(0, self.num_actions - 1) 
        #     self.rar = self.rar*self.radr 

        # else:
        self.a = np.argmax(self.Q[s,:])
            
        
        if self.verbose:  		  	   		     		  		  		    	 		 		   		 		  
            print(f"s = {s}, a = {self.a}")  		  	   		     		  		  		    	 		 		   		 		  
        return self.a  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    def query(self, s_prime, r):  		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
        Update the Q table and return an action  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
        :param s_prime: The new state  		  	   		     		  		  		    	 		 		   		 		  
        :type s_prime: int  		  	   		     		  		  		    	 		 		   		 		  
        :param r: The immediate reward  		  	   		     		  		  		    	 		 		   		 		  
        :type r: float  		  	   		     		  		  		    	 		 		   		 		  
        :return: The selected action  		  	   		     		  		  		    	 		 		   		 		  
        :rtype: int  		  	   		     		  		  		    	 		 		   		 		  
        """  
        #--------------------------------------------------Update Q-table---------------------------------------------------------------
        self.Q[self.s, self.a] = (1-self.alpha) * self.Q[self.s, self.a] + self.alpha * ( r + self.gamma*self.Q[s_prime, np.argmax(self.Q[s_prime, :])] )

        #dyna implementation process: according to each real world experiment
        if self.dyna > 0:
            #----------------------------------------------Update Model---------------------------------------------------------------
            #update T count table
            self.TC[self.s, self.a, s_prime] += 1
            #update T prob table: prob = s, a --> s', update every prob in [s,a] after each counting to make sure it have same denominator
            self.T[self.s, self.a, :] = self.TC[self.s, self.a, :]/np.sum(self.TC[self.s, self.a, :])
            #update R prob
            self.R[self.s, self.a] = (1-self.alpha)*self.R[self.s, self.a]+self.alpha*r
            """For loop is too slow to pass tests
            #----------------------------------------------Hallucination------------------------------------------------------------
            
            # for i in range(0, self.dyna):
            #     
            #     # randomly pick s and a
            #     s = rand.randint(0, 99)
            #     a = rand.randint(0, 3)
            #     # renew s and a
            #     s_new = np.argmax(self.T[s, a, :])
            #     a_new = np.argmax(self.Q[s_new, :])

            #----------------------------------------------Update Q table---------------------------------------------------------------
            #     #renew q table according above data
            #     self.Q[s,a] = (1-self.alpha)*self.Q[s,a] + self.alpha*(self.R[s,a] + self.gamma*self.Q[s_new, a_new])
            """

            """ Vectoriztion to speed up hallucination process"""
            #----------------------------------------------Hallucination------------------------------------------------------------
            Hall = np.zeros((self.dyna, 4), dtype = 'int')
            Hall[:,0] = rand.choices(range(self.num_states), k = self.dyna)
            Hall[:,1] = rand.choices(range(self.num_actions), k = self.dyna)
            Hall[:,2] = self.T[Hall[:,0], Hall[:,1],:].argmax(axis = -1)
            Hall[:,3] = self.Q[Hall[:,2], :].argmax(axis = -1)
            # Hall[:,3] = rand.choices(range(4), k = self.dyna)
            #----------------------------------------------Update Q table---------------------------------------------------------------
            self.Q[Hall[:,0],Hall[:,1]] = (1-self.alpha)*self.Q[Hall[:,0],Hall[:,1]] + self.alpha*(self.R[Hall[:,0],Hall[:,1]] + self.gamma*self.Q[Hall[:,2],Hall[:,3]])
            
        #Real world experimence based on above dyna hallucinated Q table

        #--------------------------------------------------Update Q-table---------------------------------------------------------------
        # self.Q[self.s, self.a] = (1-self.alpha) * self.Q[self.s, self.a] + self.alpha * ( r + self.gamma*self.Q[s_prime, np.argmax(self.Q[s_prime, :])] )
                
        #-----------------------------------------Generate a action according to rar and radr or argmax[Q[s, :]]-------------------------
        if rand.uniform(0.0,1.0) <= self.rar:
            action = rand.randint(0, self.num_actions - 1) 
            self.rar = self.rar*self.radr 
        else:
            action = np.argmax(self.Q[s_prime, :])

        self.s = s_prime
        self.a = action
           		     		  		  		    	 		 		   		 		  
        if self.verbose:  		  	   		     		  		  		    	 		 		   		 		  
            print(f"s = {s_prime}, a = {self.a}, r={r}")  	
        # print(self.Q)

        return self.a  		 


    #implement author func
    def author(self):
        return 'yjiang347'

  		  	   		     		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		     		  		  		    	 		 		   		 		  
    print("Remember Q from Star Trek? Well, this isn't him")  		  	   		     		  		  		    	 		 		   		 		  
    learner=QLearner(dyna=200)
    s = 75 # our initial state
    a = learner.querysetstate(s) # action for state s
    # the new state we end up in after taking action a in state s
    if a == 0:
        s_prime = s-10
    elif a == 1:
        s_prime = s+1
    elif a == 2:
        s_prime = s+10
    elif a == 3:
        s_prime = s-1
    r = -1 # reward for taking action a in state s
    next_action = learner.query(s_prime, r)