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
	double[] split_points;
	TreeNode[] childrenNodes;
}

public class DecisionTree extends Classifier {

    private boolean _isClassification;
    private double[][] _features;
    private boolean[] _isCategory;
    private double[] _labels;
    private double[] _defaults;
    
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
        
        //处理缺失属性
        _defaults = kill_missing_data();
        
    	root = build_decision_tree(set, attr_index);
    }
    
    private double[] kill_missing_data() {
    	int num = _isCategory.length - 1;
    	double[] defaults = new double[num];
    	
    	for (int i = 0; i < defaults.length; ++i) {
    		if (_isCategory[i]) {
    			//离散属性取最多的值
    	    	HashMap<Double, Integer> counter = new HashMap<Double, Integer>();
    	    	for (int j = 0; j < _features.length; ++j) {
    	    		double feature = _features[j][i];
    	    		if (!Double.isNaN(feature)) {
    	    			if (counter.get(feature) == null) {
        	    			counter.put(feature, 1);
        	    		} else {
        	    			int count = counter.get(feature) + 1;
        	    			counter.put(feature, count);
        	    		}
    	    		}
    	    	}
    	    	
    	    	int max_time = 0;
    	    	double value = 0;
    	    	Iterator<Double> iterator = counter.keySet().iterator();
    	    	while (iterator.hasNext()) {
    	    		double key = iterator.next();
    	    		int count = counter.get(key);
    	    		if (count > max_time) {
    	    			max_time = count;
    	    			value = key;
    	    		}
    	    	}
    	    	defaults[i] = value;
    		} else {
    			//连续属性取平均值
    			int count = 0;
    			double total = 0;
    			for (int j = 0; j < _features.length; ++j) {
    				if (!Double.isNaN(_features[j][i])) {
    					count++;
    					total += _features[j][i];
    				}
    			}
    			defaults[i] = total / count;
    		}
    	}
    	
    	//代换
    	for (int i = 0; i < _features.length; ++i) {
    		for (int j = 0; j < defaults.length; ++j) {
    			if (Double.isNaN(_features[i][j])) {
    				_features[i][j] = defaults[j];
    			}
    		}
    	}
    	return defaults;
    }

    @Override
    public double predict(double[] features) {
    	//处理缺失属性
    	for (int i = 0; i < features.length; ++i) {
    		if (Double.isNaN(features[i])) {
    			features[i] = _defaults[i];
    		}
    	}
    	
        double anser = predict_with_decision_tree(features, root);
        //System.out.println(anser);
        return anser;
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
        		return node.label;
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
    			break;
    		}
    	}
    	if (flag) {
    		node.label = label;
    		return node;
    	}
    	
    	//没有可用属性标记为大多数(离散)或平均值(连续)
    	if (_isClassification) {
			node.label = most_label(set);
		} else {
			node.label = mean_value(set);
		}
    	if (node.attr_index == null || node.attr_index.length == 0) {
    		return node;
    	}
    	
    	//寻找最优切割属性
    	node.split_attr = attribute_selection(node);
    	//没有可以分割的属性
    	if (node.split_attr < 0) {
    		return node;
    	}
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
    
    private double mean_value(int[] set) {
    	double temp = 0;
    	for (int index : set) {
    		temp += _labels[index];
    	}
    	return temp / set.length;
    }
    
    private int attribute_selection(TreeNode node) {
    	double reference_value = _isClassification ? 0 : -1;
    	int selected_attribute = -1;
    	if (_isClassification) {
        	for (int attribute : node.attr_index) {
        		double temp = gain_ratio_use_discrete_attribute(node.set, attribute);
        		if (temp > reference_value) {
        			reference_value = temp;
        			selected_attribute = attribute;
        		}
        	}
    	} else {
        	for (int attribute : node.attr_index) {
        		double temp = mse_use_attribute(node.set, attribute);
        		if (reference_value < 0 || temp < reference_value) {
        			reference_value = temp;
        			selected_attribute = attribute;
        		}
        	}
    	}
    	return selected_attribute;
    }
    
    //增益率 C4.5
    private double gain_ratio_use_discrete_attribute(int[] set, int attribute) {
    	double entropy_before_split = entropy(set);
    	
    	double entropy_after_split = 0;
    	double split_information = 0;
    	for (int[] sub_set : split_with_attribute(set, attribute, null)) {
    		entropy_after_split += (double)sub_set.length / set.length * entropy(sub_set);
    		double p = (double)sub_set.length / set.length;
    		split_information += - p * Math.log(p);
    	}
    	
    	return (entropy_before_split - entropy_after_split) / split_information;
    }
    
    private double gain_ratio_use_numerical_attribute(int[] set, int attribute, int[] part_a, int[] part_b) {
    	double entropy_before_split = entropy(set);
    	double entropy_after_split = (part_a.length * entropy(part_a) + part_b.length * entropy(part_b)) / set.length;
    	
    	double split_information = 0;
    	double p = (double)part_a.length / set.length;
    	split_information += - p * Math.log(p);
    	p = (double)part_b.length / set.length;
    	split_information += - p * Math.log(p);
    	
    	return (entropy_before_split - entropy_after_split) / split_information;
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
    		int[] ordered_set = set.clone();
            for (int i = 0; i < ordered_set.length; i++) {
                for (int j = i + 1; j < ordered_set.length; j++) {
                    if (_features[ordered_set[i]][attribute] > _features[ordered_set[j]][attribute]) {
                        int temp = ordered_set[i];
                        ordered_set[i] = ordered_set[j];
                        ordered_set[j] = temp;
                    }
                }
            }
            
            double reference_value = _isClassification ? 0 : -1;
            double best_split_point = 0;
            int[][] result = new int[2][];
            for (int i = 0; i < ordered_set.length - 1; ++i) {
                double split_point = (_features[ordered_set[i]][attribute] + _features[ordered_set[i + 1]][attribute]) / 2;
                int[] sub_set_a = new int[i + 1];
                int[] sub_set_b = new int[set.length - i - 1];
                
                for (int j = 0; j < sub_set_a.length; ++j) {
                    sub_set_a[j] = ordered_set[j];
                }
                for (int j = 0; j < sub_set_b.length; ++j) {
                    sub_set_b[j] = ordered_set[j + i + 1];
                }
    			
    			if (_isClassification) {
    				double temp = gain_ratio_use_numerical_attribute(set, attribute, sub_set_a, sub_set_b);
        			if (temp > reference_value) {
        				reference_value = temp;
        				best_split_point = split_point;
        				result[0] = sub_set_a;
        				result[1] = sub_set_b;
        			}
    			} else {
    				double temp = (sub_set_a.length * mse(sub_set_a) + sub_set_b.length * mse(sub_set_b)) / set.length;
    				if (reference_value < 0 || temp < reference_value) {
    					reference_value = temp;
    					best_split_point = split_point;
    					result[0] = sub_set_a;
    					result[1] = sub_set_b;
    				}
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
    
    private double mse(int[] set) {
    	double mean = mean_value(set);
    	
    	double temp = 0;
    	for (int index : set) {
    		double t = _labels[index] - mean;
    		temp += t * t;
    	}
    	return temp / set.length;
    }
    
    private double mse_use_attribute(int[] set, int attribute) {
    	double mse = 0;
    	for (int[] sub_set : split_with_attribute(set, attribute, null)) {
    		mse += (double)sub_set.length / set.length * mse(sub_set);
    	}
    	return mse;
    }
    
}
