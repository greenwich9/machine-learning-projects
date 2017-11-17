import sys
import scipy.io.arff as sciarff
import numpy as np
import random
import math
import pylab


# calculate the entropy of classes
def entropy(clas):
    n = len(clas)
    count = {}
    for label in labels:
        count[label] = 0
    for cls in clas:
        count[cls] += 1
    if count[labels[0]] == 0 or count[labels[0]] == n:
        return 0 ,count

    else:
        etp = [-count[i]/float(n)*np.log2(count[i]/float(n)) for i in labels]

        return float(sum(etp)), count


  #numeric features
def num_split(x, cls):
	xsort = np.sort(x)
	n = len(cls)
	min_etp = 1
	etp = -1
	final_threshold = -1
	for i in range(1, n):
		if xsort[i - 1] < xsort[i]:
			threshold = (float(xsort[i - 1]) + float(xsort[i]))/2
			left= [cls[j] for j in range(n) if float(x[j]) <= threshold]
			right = [cls[j] for j in range(n) if float(x[j]) > threshold]
			etp = (i*entropy(left)[0] + (n - i) * entropy(right)[0])/n
			if etp < min_etp:
				min_etp = etp
				final_threshold = threshold
	return final_threshold, float(min_etp)
    

def cond_entropy(i, features, cls):
  	n = len(cls)
  	etp = [sum(features == j)*entropy(cls[j == features])[0] for j in classes[i][1]]
  	return float(sum(etp)/n)

#check if the data has the same feature
def same_featrue(dataset):
    idx = [min(dataset[0] == dataset[i]) for i in range(len(dataset))]
    if min(idx) == True:
        return True
    return False

#check if the sameples are of the same group
def same_gp(dataset):
    cls = dataset[:, -1]
    if len(cls) == 0:
    	return True
    s = cls[0]
    for i in range(len(cls)):
    	if cls[i] != s:
    		return False
    return True


# build the tree using m and trainning dataset, take the plurality of parent in the function
def build_tree(dataset, m, plurality):
	if len(dataset) == 0:
		return None
	else:
		etp, count = entropy(dataset[:,-1])
		c0 = {}
    	for label in labels:
        	c0[label] = 0
      	for i in count:
      		if count[i] > count[plurality]:
      			plurality = i
      	if same_gp(dataset) or same_featrue(dataset[:,:-1]) or len(dataset) < m:
      		return {'status' : 'Terminate', 'prediction' : plurality, 'count' : count}

      	else:
      		final_idx = -1
      		max_gain = -1
      		for i in range(len(classes) - 1):
      			if classes[i][0] == 'numeric':
      				dataset_float = np.array(dataset[:,i], dtype = 'float64')
      				
      				thres, cond_etp = num_split(dataset_float, dataset[:, -1])
      				gain = etp - cond_etp
      				if float(gain) > float(max_gain):
      					max_gain = float(gain)
      					final_idx = i
      					threshold = thres
      			elif classes[i][0] == 'nominal':
      				features = dataset[:, i]
      				cond_etp = cond_entropy(i, features, dataset[:, -1])
      				gain = etp - cond_etp
      				if gain > max_gain:
      					final_idx = i
      					max_gain = gain
      	if max_gain <= 0:
      		return {'status' : 'Terminate', 'prediction' : plurality, 'count' : count}

      	if classes[final_idx][0] == 'nominal':
      		
      		subtree = []
      		for att in classes[final_idx][1]:
      			branch = build_tree(dataset[dataset[:,final_idx] == att], m, plurality)
      			if branch == None:
      				branch = {'status' : 'Terminate', 'prediction' : plurality, 'count' : count}
      			subtree.append(branch)
      		return {'status' : 'Continue', 'threshold' : None, 'count' : count, 'subtree' : subtree, 'index' : final_idx}

      	elif classes[final_idx][0] == 'numeric':
      		float_data = np.array(dataset[:, final_idx], dtype = 'float64')
      		left = build_tree(dataset[float_data <= threshold], m, plurality)
      		if left == None:
      			left = {'status' : 'Terminate', 'prediction' : plurality, 'count' : count}
      		right = build_tree(dataset[float_data > threshold], m, plurality)
      		if right == None:
      			right = {'status' : 'Terminate', 'prediction' : plurality, 'count' : count}
      		return {'status' : 'Continue', 'threshold' : threshold, 'count' : count, 'subtree' : [left, right], 'index' : final_idx}

# predict for a certain sample
def predict(x, tree):
	if tree['status'] == 'Terminate':
		return tree['prediction']
	else:
		loc = tree['index']
		if classes[loc][0] == 'numeric':
			if float(x[loc]) <= tree['threshold']:
				clslabel = predict(x, tree['subtree'][0])
			else:
				clslabel = predict(x, tree['subtree'][1])
			return clslabel

		else:
			nbranch = len(classes[loc][1])
			for i in range(nbranch):
				if classes[loc][1][i] == x[loc]:
					idx = i
					break
			clslabel = predict(x, tree['subtree'][idx])
			return clslabel

# calculate and print the results for the dataset
def test_tree(dataset, dtree, display = True):
    num_test = dataset.shape[0]
    num_correct = 0
    if display:
        print "<Predictions for the Test Set Instances>"
    for i in xrange(num_test): 
        y_pred = predict(dataset[i,:-1], dtree)
        if display:
            print('{0}: Actual: {1} Predicted: {2}'.format(i+1, dataset[i,-1], y_pred))
        if y_pred == dataset[i,-1]:
            num_correct += 1
    if display: 
        print (" Number of correctly classified: {0} Total number of test instances: {1}".format(num_correct, num_test) )

    return num_correct, num_test  


def print_tree(dtree, level = 0):
    indent = ('|'+' '*8)*level
    if tree['status'] != 'Terminate': 
        # nominal    
        if dtree['threshold'] == None:
            for i in range(len(classes[dtree['index']][1])): 
                print indent + features[dtree['index']] + ' = ' + classes[dtree['index']][1][i],   
                if dtree['subtree'][i]['status'] == 'Terminate':      
                    try:
                        print ' ', dtree['subtree'][i]['count'] , ': ' + dtree['subtree'][i]['prediction'] 
                                
                    except:
                        print  ' : ' + dtree['subtree'][i]['prediction']
                else:
                    print ' ', dtree['subtree'][i]['count']
                    print_tree(dtree['subtree'][i], level+1)

                
        # numeric attributes
        else: 
            print indent + features[dtree['index']] + ' <= ' + str(dtree['threshold']),    
            if dtree['subtree'][0]['status'] == 'Terminate':  
                try: 
                    print ' ', dtree['subtree'][0]['count'], ': ' + dtree['subtree'][0]['prediction']
                                
                except:
                    print  ' : ' + dtree['subtree'][0]['prediction']
            else:
                print ' ', dtree['subtree'][0]['count']
                
                print_tree(dtree['subtree'][0], level+1)

            
            print indent + features[dtree['index']] + ' > ' + str(dtree['threshold']),
            if dtree['subtree'][1]['status'] == 'Terminate':       
                try: 
                    print ' ', dtree['subtree'][1]['count'], ': ' + dtree['subtree'][1]['prediction']
                                
                except:
                    print  ' : ' + dtree['subtree'][1]['prediction']
            else:
                print ' ', dtree['subtree'][1]['count']      
                print_tree(dtree['subtree'][1], level + 1) 
    else:
        pass
                     
def print_learning_curve(train, test, fraction, plurality, times = 10, m = 4):
    num_train = train.shape[0]
    num_test = test.shape[0]
    table = np.zeros([len(fraction),4])
    for i in xrange(len(fraction)):
        num_correct = []
        if fraction[i] == 1:
            tree = build_tree(train, m, plurality)
            num_correct.append(test_tree(test, tree, False)[0])           
        else: 
            for t in xrange(times):
                random.seed(t)
                inds = random.sample(range(num_train), int(math.ceil(fraction[i]*num_train)))

                trainSampled = train[inds] 
                tree = build_tree(trainSampled, m, plurality)
                num_correct.append(test_tree(test, tree, False)[0])
        table[i,:] = np.array([fraction[i], min(num_correct)/float(num_test), np.mean(num_correct)/float(num_test), max(num_correct)/float(num_test)])
    pylab.figure(1)
    pylab.plot(table[:,0], table[:,1], 'rx')
    pylab.plot(table[:,0], table[:,1], 'r', label = 'minimum')
    pylab.hold(True)
    pylab.plot(table[:,0], table[:,2], 'gx')
    pylab.plot(table[:,0], table[:,2], 'g', label = 'average')
    pylab.hold(True)
    pylab.plot(table[:,0], table[:,3], 'bx')
    pylab.plot(table[:,0], table[:,3], 'b', label = 'maximum')
    
    pylab.title("Accuracies for different sample sizes\n (m = %d)" %m)
    pylab.xlabel("Number of training samples in percentage")
    pylab.ylabel("test-set accuracy")
    pylab.legend(loc = 'lower right')

    
def print_size(train, test, m_list, plurality):

    accuracy = []
    for m in m_list:
        tree = build_tree(train, m, plurality)
        num_correct, num_test = test_tree(test, tree, False)
        accuracy.append(num_correct/float(num_test))
    pylab.figure(2)
    pylab.plot(m_list, accuracy, 'bx')
    pylab.plot(m_list, accuracy, 'r')
    pylab.title("Accuracy for different m")
    pylab.xlabel("m")
    pylab.ylabel("Accuracy")

args = [arg for arg in sys.argv]
train = args[1]
test = args[2] 
m = int(args[3])

## load data
train_data = sciarff.loadarff(train)
test_data = sciarff.loadarff(test)  

## reshape the datasets
train = np.array([[i for i in train_data[0][j]] for j in range(train_data[0].shape[0])])
test = np.array([[i for i in test_data[0][j]] for j in range(test_data[0].shape[0])])

## get the feature names and the class names
features = train_data[1].names()
classes = [train_data[1][feat] for feat in train_data[1].names()]
labels = classes[-1][1]

num_feat = len(features)-1
nLabels = len(labels)

plurality = 'negative'

tree = build_tree(train, m, plurality)


# plot tree
print_tree(tree, 0)

#print the prediction
test_tree(test, tree)

# plot accuracy curves
runtimes = 10
ratio = [0.05, 0.1, 0.2, 0.5, 1]
print_learning_curve(train, test, ratio, plurality, runtimes)

# plot test-set accuracy against different m
mlist = [2, 5, 10, 20]   
print_size(train, test, mlist, plurality)

pylab.show()



