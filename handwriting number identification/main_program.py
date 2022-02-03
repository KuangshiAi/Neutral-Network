import numpy
import scipy.special
import matplotlib.pyplot
import math
inputnodes=784
hiddennodes=100
outputnodes=10
learningrate=0.3
class neuralNetwork:
    def _init_(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inode=inputnodes
        self.hnode=hiddennodes
        self.onode=outputnodes
        self.lr=learningrate
        self.wih=numpy.random.normal(0.0,pow(self.hnode,-0.5),(self.hnode,self.inode))
        self.who=numpy.random.normal(0.0,pow(self.onode,-0.5),(self.onode,self.hnode))
        self.activationfunction=lambda x:scipy.special.expit(x)
        pass
    def query(self,inputs_list):
        #默认情况下array生成向量是横向的，故需要转置；ndmin设置最小维度
        inputs=numpy.array(inputs_list,ndmin=2).T
        hidden_inputs=numpy.dot(self.wih,inputs)
        hidden_outputs=self.activationfunction(hidden_inputs)
        final_inputs=numpy.dot(self.who,hidden_outputs)
        final_outputs=self.activationfunction(final_inputs)
        return final_outputs
    def train(self,inputs_list,targets_list):
        targets=numpy.array(targets_list,ndmin=2).T
        inputs=numpy.array(inputs_list,ndmin=2).T
        hidden_inputs=numpy.dot(self.wih,inputs)
        hidden_outputs=self.activationfunction(hidden_inputs)
        final_inputs=numpy.dot(self.who,hidden_outputs)
        final_outputs=self.activationfunction(final_inputs)
        output_errors=targets-final_outputs
        hidden_errors=numpy.dot(self.who.T,output_errors)
        self.who+=self.lr*numpy.dot(output_errors*(1.0-final_outputs)*final_outputs,hidden_outputs.T)
        self.wih+=self.lr*numpy.dot(hidden_errors*(1.0-hidden_outputs)*hidden_outputs,inputs.T)
        pass
proj= neuralNetwork ()
proj._init_(inputnodes,hiddennodes,outputnodes,learningrate)
train_file=open(r'C:\Users\艾志敏\Desktop\程序设计\Python\mnist_250.csv','r')
train_list=train_file.readlines()
train_file.close()
for record in train_list:
    all_value=record.split(',')
    input=(numpy.asfarray(all_value[1:])/255.0*0.99)+0.01
    target=numpy.zeros(outputnodes)+0.01
    target[int(all_value[0])]=0.99
    proj.train(input,target)
f=open(r'C:\Users\艾志敏\Desktop\程序设计\Python\mnist_10_test.csv','r')
test_list=f.readlines()
f.close()
scorecard=[]
epochs=2
for e in range(epochs):
    for record in test_list:
        all_value=record.split(',')
        input=(numpy.asfarray(all_value[1:])/255.0*0.99)+0.01
        output=proj.query(input)
        label=numpy.argmax(output)
        correct_label=all_value[0]
        if int(label)==int(correct_label):
            scorecard.append(1)
        else:
            scorecard.append(0)
        print("correct label is %d"%int(correct_label))
        print("neural network's answer is %d"%int(label))
scorecard_array=numpy.asarray(scorecard)
print("performance=",scorecard_array.sum()/scorecard_array.size)
