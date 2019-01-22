import numpy as np
from modify_xml import modify_xml

class GMM:
    def __init__(self, robot_name='ant', m = 100, learning_rate=0.01, params_mu = None, params_sig = None):
        self.robot_name = robot_name
        if (robot_name == 'ant') and (params_mu == None):
            params_mu = np.array([0.2, 0.08, 0.4, 0.08])
        if (robot_name == 'hopper') and (params_mu == None):
            params_mu = np.array([0.4, 0.05, 0.45, 0.05, 0.5, 0.04, 0.13, 0.06, 0.26, 0.06]) # length and rad for torso, thigh, leg, foot1, foot2
        if (robot_name == 'walker2d') and (params_mu == None):
            params_mu = np.array([0.4, 0.05, 0.45, 0.05, 0.5, 0.04, 0.2, 0.06]) # length and rad for torso, thigh, leg, foot
        self.params_mu = params_mu
        self.params_sig = self.params_mu/3.0
        self.num_params = len (self.params_mu)
        self.N = m
        self.P = self.params_mu.shape[0]
        self.alpha = learning_rate
        self.T = np.zeros((self.P,self.N))
        self.S = np.zeros((self.P,self.N))
        self.base = 0
        self.NofEpisodes = 1.0
        self.sig_min = 0.00001


    def sample(self, m=None, to_update = False):
        self.params_sig_copy = np.maximum(self.params_sig, self.sig_min)
        if m == None:
            m = self.N
        params_list = []
        for i in range(m):
            deltas = []
            for mu, sig in zip(self.params_mu, self.params_sig_copy):
                delta = np.random.normal(0, sig)
                while mu - abs(delta) < 0:
                    delta = np.random.normal(mu, sig)
                deltas.append(delta)
            deltas = np.array(deltas)
            params1 = self.params_mu + deltas
            params2 = self.params_mu - deltas
            params_list += [params1, params2]
            if to_update:
                self.T[:,i] = deltas
                self.S[:,i] = (deltas**2-self.params_sig_copy**2)/self.params_sig_copy

        return params_list

    def update(self,rewardsplus, rewardsminus):
        r_T = (rewardsplus-rewardsminus).reshape((-1,1))
        r_S = ((rewardsplus+rewardsminus)/2-self.base).reshape((-1,1))

        # print("mu:", self.params_mu)
        # print("sig:", self.params_sig)
        # print("mu_grad:", self.T.dot(r_T).ravel())
        # print("sig_grad:", self.S.dot(r_S).ravel())
        # mu = self.params_mu
        # sig = self.params_sig
        # mu_grad = self.T.dot(r_T).ravel()
        # sig_grad = self.S.dot(r_S).ravel()
        
        # mu_min_idx = np.argmax(np.absolute(np.divide(mu, mu_grad)))
        # sig_min_idx = np.argmax(np.absolute(np.divide(sig, sig_grad)))

        # print(mu[mu_min_idx], self.T.dot(r_T).ravel()[mu_min_idx], np.max(np.absolute(np.divide(mu, mu_grad))))
        # print(sig[sig_min_idx], self.S.dot(r_S).ravel()[sig_min_idx], np.max(np.absolute(np.divide(sig, sig_grad))))



        self.params_mu += self.alpha*self.T.dot(r_T).ravel()
        self.params_sig += self.alpha*self.S.dot(r_S).ravel()
        for r in rewardsplus:
            self.base += 1/self.NofEpisodes*(r-self.base)
            self.NofEpisodes += 1
        for r in rewardsminus:
            self.base += 1/self.NofEpisodes*(r-self.base)
            self.NofEpisodes += 1

    def modify_file(self, params):
        if self.robot_name == 'ant':
            if type(params) == list:
                modify_xml(self.robot_name, parameters = params*4)
            else:
                modify_xml(self.robot_name, parameters = params.tolist()*4)
        else:
            if type(params) == list:
                modify_xml(self.robot_name, parameters = params)
            else:
                modify_xml(self.robot_name, parameters = params.tolist())


def main():
    myGMM = GMM('hopper',m=3)   
    params_list = myGMM.sample(3)
    print (len(params_list))
    myGMM.update(np.ones(3),-np.ones(3))
    myGMM.modify_file(params_list[0])

if __name__ == '__main__':
    main()
