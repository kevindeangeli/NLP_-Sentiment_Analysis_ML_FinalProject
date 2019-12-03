'''
Created by Kevin De Angeli
Date: 2019-12-02
'''

import numpy as np
from Distances import *

class mpp:
   def __init__(self, case=1):
      # init prior probability, equal distribution
      # self.classn = len(self.classes)
      # self.pw = np.full(self.classn, 1/self.classn)

      # self.covs, self.means, self.covavg, self.varavg = \
      #     self.train(self.train_data, self.classes)
      self.case_ = case
      self.pw_ = None

   def fit(self, Tr, y, fX=False):
      # derive the model
      self.covs_, self.means_ = {}, {}
      self.covsum_ = None

      self.classes_ = np.unique(y)  # get unique labels as dictionary items
      self.classn_ = len(self.classes_)

      for c in self.classes_:
         arr = Tr[y == c]
         self.covs_[c] = np.cov(np.transpose(arr))
         self.means_[c] = np.mean(arr, axis=0)  # mean along rows
         if self.covsum_ is None:
            self.covsum_ = self.covs_[c]
         else:
            self.covsum_ += self.covs_[c]

      if fX == False:
         # used by case II
         self.covavg_ = self.covsum_ / self.classn_

         # used by case I
         self.varavg_ = np.sum(np.diagonal(self.covavg_)) / len(self.classes_)
      else:
         self.covavg_ = np.std(Tr)
         self.varavg = np.var(Tr)

   def predict(self, T):
      # eval all data
      y = []
      disc = np.zeros(self.classn_)
      nr, _ = T.shape

      if self.pw_ is None:
         self.pw_ = np.full(self.classn_, 1 / self.classn_)

      for i in range(nr):
         for c in self.classes_:
            if self.case_ == 1:
               edist2 = euc2(self.means_[c], T[i])
               disc[c] = -edist2 / (2 * self.varavg_) + np.log(self.pw_[c])
            elif self.case_ == 2:
               mdist2 = mah2(self.means_[c], T[i], self.covavg_)
               disc[c] = -mdist2 / 2 + np.log(self.pw_[c])
            elif self.case_ == 3:
               mdist2 = mah2(self.means_[c], T[i], self.covs_[c])
               disc[c] = -mdist2 / 2 - np.log(np.linalg.det(self.covs_[c])) / 2 \
                         + np.log(self.pw_[c])
            else:
               print("Can only handle case numbers 1, 2, 3.")
               sys.exit(1)
         y.append(disc.argmax())

      return y

