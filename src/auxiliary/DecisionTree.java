package auxiliary;

import java.util.*;

/**
 *
 * @author MF1333020 孔繁宇
 */

//决策树节点结构
class TreeNode {
	int[] set;
	int[] attr_index;
	double label;
	int split_attr;
	TreeNode[] childrenNodes;
	double[] split_points;
}

public class DecisionTree extends Classifier {

    private boolean _isClassification;
    private double[][] _features;
    private boolean[] _isCategory;
    private double[] _labels;
    
    private TreeNode root;

    public DecisionTree() {
    }
    
    @Override
    public void train(boolean[] isCategory, double[][] features, double[] labels) {
        _isClassification = isCategory[isCategory.length - 1];
        _features = features;
        _isCategory = isCategory;
        _labels = labels;
        
        int set[] = new int[_features.length];
        for (int i = 0; i < set.length; ++i) {
        	set[i] = i;
        }
        
        int attr_index[] = new int[_features[0].length];
        for (int i = 0; i < attr_index.length; ++i) {
        	attr_index[i] = i;
        }
        
        
    	root = build_decision_tree(set, attr_index);
    }

    @Override
    public double predict(double[] features) {
        return predict_with_decision_tree(features, root);
    }
    
    private double predict_with_decision_tree(double[] features, TreeNode node) {
    	if (node.childrenNodes == null) {
    		return node.label;
    	}
    	
    	double feature = features[node.split_attr];
    	
    	if (_isCategory[node.split_attr]) {
    		//离散属性
    		int branch = -1;
        	for (int i = 0; i < node.split_points.length; ++i) {
        		if (node.split_points[i] == feature) {
        			branch = i;
        			break;
        		}
        	}
        	
        	if (branch < 0) {
        		return 0;
        	} else {
        		return predict_with_decision_tree(features, node.childrenNodes[branch]);
        	}
    	} else {
    		//连续属性
    		if (feature < node.split_points[0]) {
    			return predict_with_decision_tree(features, node.childrenNodes[0]);
    		} else {
    			return predict_with_decision_tree(features, node.childrenNodes[1]);
    		}
    	}
    	
    }
    
    private TreeNode build_decision_tree(int[] set, int[] attr_index) {
    	TreeNode node = new TreeNode();
    	node.set = set;
    	node.attr_index = attr_index;
    	node.label = 0;
    	node.childrenNodes = null;
    	
    	//都为同类返回直接返回
    	double label = _labels[node.set[0]];
    	boolean flag = true;
    	for (int i = 0; i < node.set.length; ++i) {
    		if (_labels[node.set[i]] != label) {
    			flag = false;
    		}
    	}
    	if (flag) {
    		node.label = label;
    		return node;
    	}
    	
    	//没有可用属性标记为大多数
    	if (node.attr_index == null || node.attr_index.length == 0) {
    		node.label = most_label(set);
    		return node;
    	}
    	
    	//寻找最优切割属性
    	node.split_attr = attribute_selection(node);
    	//System.out.println(node.split_attr);
    	int[][] sub_sets = split_with_attribute(node.set, node.split_attr, node);
    	
    	//去掉已使用的属性
    	int[] child_attr_index = new int[attr_index.length - 1];
    	int t = 0;
    	for (int index : attr_index) {
    		if (index != node.split_attr) {
    			child_attr_index[t++] = index;
    		}
    	}
    	
    	//递归建立子节点
    	node.childrenNodes = new TreeNode[sub_sets.length];
    	for (int i = 0; i < sub_sets.length; ++i) {
    		node.childrenNodes[i] = build_decision_tree(sub_sets[i], child_attr_index);
    	}
    	
    	return node;
    }
    
    private double most_label(int[] set) {
    	HashMap<Double, Integer> counter = new HashMap<Double, Integer>();
    	for (int item : set) {
    		double label = _labels[item];
    		if (counter.get(label) == null) {
    			counter.put(label, 1);
    		} else {
    			int count = counter.get(label) + 1;
    			counter.put(label, count);
    		}
    	}
    	
    	int max_time = 0;
    	double label = 0;
    	Iterator<Double> iterator = counter.keySet().iterator();
    	while (iterator.hasNext()) {
    		double key = iterator.next();
    		int count = counter.get(key);
    		if (count > max_time) {
    			max_time = count;
    			label = key;
    		}
    	}
    	return label;
    }
    
    private int attribute_selection(TreeNode node) {
    	double information_gain = -1;
    	int max_attribute = -1;
    	for (int attribute : node.attr_index) {
    		double temp = information_gain_use_attribute(node.set, attribute);
    		//System.out.println(temp);
    		if (temp > information_gain) {
    			information_gain = temp;
    			max_attribute = attribute;
    		}
    	}
    	return max_attribute;
    }
    
    private double information_gain_use_attribute(int[] set, int attribute) {
    	double entropy_before_split = entropy(set);
    	
    	double entropy_after_split = 0;
    	for (int[] sub_set : split_with_attribute(set, attribute, null)) {
    		entropy_after_split += (double)sub_set.length / set.length * entropy(sub_set);
    	}
    	
    	//System.out.printf("%f %f\n", entropy_before_split, entropy_after_split);
    	return entropy_before_split - entropy_after_split;
    }
    
    private int[][] split_with_attribute(int[] set, int attribute, TreeNode node) {
    	if (_isCategory[attribute]) {
    		//离散属性
    		int amount_of_features = 0;
    		HashMap<Double, Integer> counter = new HashMap<Double, Integer>();
    		HashMap<Double, Integer> index_recorder = new HashMap<Double, Integer>();
        	for (int item : set) {
        		double feature = _features[item][attribute];
        		if (counter.get(feature) == null) {
        			counter.put(feature, 1);
        			index_recorder.put(feature, amount_of_features++);
        		} else {
        			int count = counter.get(feature) + 1;
        			counter.put(feature, count);
        		}
        	}
        	
        	//记录切割点
        	if (node != null) {
        		node.split_points = new double[amount_of_features];
        		Iterator<Double> iterator = index_recorder.keySet().iterator();
        		
        		while (iterator.hasNext()) {
        			double key = iterator.next();
        			int value = index_recorder.get(key);
        			node.split_points[value] = key;
        		}
        	}
        	
        	int[][] result = new int[amount_of_features][];
        	int[] t_index = new int[amount_of_features];
        	for (int i = 0; i < amount_of_features; ++i) t_index[i] = 0;
        	
        	for (int item : set) {
        		int index = index_recorder.get(_features[item][attribute]);
        		if (result[index] == null) {
        			result[index] = new int[counter.get(_features[item][attribute])];
        		}
        		result[index][t_index[index]++] = item;
        	}
        	
    		return result;
    	} else {
    		//连续属性
    		double[] points = new double[set.length];
    		for (int i = 0; i < set.length; ++i) {
    			points[i] = _features[set[i]][attribute];
    		}
    		Arrays.sort(points);
    		
    		double entropy = -1;
    		double best_split_point = 0;
    		int[][] result = new int[2][];
    		//for (int i = 0; i < points.length - 1; ++i) {
    		for (int i = points.length / 2; i < points.length / 2 + 1; ++i) {
    			double split_point = (points[i] + points[i + 1]) / 2;
    			int[] sub_set_a = new int[i + 1];
    			int[] sub_set_b = new int[set.length - i - 1];
    			
    			for (int j = 0; j < sub_set_a.length; ++j) {
    				sub_set_a[j] = set[j];
    			}
    			for (int j = 0; j < sub_set_b.length; ++j) {
    				sub_set_b[j] = set[j + i + 1];
    			}
    			
    			double temp = ((double)sub_set_a.length / set.length) * entropy(sub_set_a)
    					+ ((double)sub_set_b.length / set.length) * entropy(sub_set_b);
    			if (entropy < 0 || temp < entropy) {
    				entropy = temp;
    				best_split_point = split_point;
    				result[0] = sub_set_a;
    				result[1] = sub_set_b;
    			}
    		}
    		if (node != null) {
    			node.split_points = new double[1];
    			node.split_points[0] = best_split_point;
    		}
    		return result;
    	}
    }
    
    private double entropy(int[] set) {
    	HashMap<Double, Integer> counter = new HashMap<Double, Integer>();
    	for (int item : set) {
    		double label = _labels[item];
    		if (counter.get(label) == null) {
    			counter.put(label, 1);
    		} else {
    			int count = counter.get(label) + 1;
    			counter.put(label, count);
    		}
    	}
    	
    	double result = 0;
    	Iterator<Double> iterator = counter.keySet().iterator();
    	while (iterator.hasNext()) {
    		int count = counter.get(iterator.next());
    		double p = (double)count / set.length;
    		result += - p * Math.log(p);
    	}
    	
    	return result;
    }
    
}
