#!/usr/bin/python
#-*- coding: utf-8
__author__ = 'livvddcc@gmail.com'

import numpy as np

class BPNN:
    def __init__(self, input_array, output_array, layer_node_count_list, w_step_size=0.02, b_step_size=0.02):
        
        self.layers = []
        self.input_array = input_array
        self.output_array = output_array
        self.layer_node_count_list = layer_node_count_list
        self.w_step_size = w_step_size
        self.b_step_size = b_step_size
        
        self.layer_count = len(self.layer_node_count_list) - 1
        self.w_dic = {}
        
        self.b_dic = {}
        
        for i in range(1,len(self.layer_node_count_list)):
            self.w_dic[i] = 2*(np.random.random((layer_node_count_list[i],layer_node_count_list[i-1]))-0.5)
            self.b_dic[i] = 2*(np.random.random((1, layer_node_count_list[i]))-0.5)

    def sigmoid(self, x, derivative=False):
        if derivative:
            return x*(1-x)
        else:
            return 1/(1+np.exp(-x))

    def forward(self):
        self.layers = []
        self.layers.append(self.input_array)
        for i in range(self.layer_count):
            z = np.dot(self.layers[i], self.w_dic[i+1].T) + self.b_dic[i+1]
            a = self.sigmoid(z)
            self.layers.append(a)

    def backward(self):
        delta_list = []
        theta_output = (self.output_array - self.layers[-1]) * self.sigmoid(self.layers[-1],derivative=True)
                    
        delta_list.append(theta_output)
        for i in range(self.layer_count-1, 0, -1):
            theta = np.dot(delta_list[-1], self.w_dic[i+1]) * self.sigmoid(self.layers[i],derivative=True)
                     
            delta_list.append(theta)

        delta_list.reverse()
        w_change_dic = {}
        b_change_dic = {}
        N = len(self.input_array)
        for i in range(len(delta_list)):
            w_change_dic[i+1] = np.dot(delta_list[i].T,self.layers[i]) / float(N) * self.w_step_size
            b_change_dic[i+1] = np.sum(delta_list[i],0)/float(N)*self.b_step_size

        for i in w_change_dic.keys():
            self.w_dic[i] += w_change_dic[i]
            self.b_dic[i] += b_change_dic[i]     



