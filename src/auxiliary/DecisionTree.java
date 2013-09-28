package auxiliary;

import java.util.*;

/**
 *
 * @author MF1333020 �׷���
 */

//�������ڵ�ṹ
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
        
        //����ȱʧ����
        _defaults = kill_missing_data();
        
    	root = build_decision_tree(set, attr_index);
    }
    
    public double[] kill_missing_data() {
    	int num = _isCategory.length - 1;
    	double[] defaults = new double[num];
    	
    	for (int i = 0; i < defaults.length; ++i) {
    		if (_isCategory[i]) {
    			//��ɢ����ȡ����ֵ
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
    			//��������ȡƽ��ֵ
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
    	
    	//����
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
    	//����ȱʧ����
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
    		//��ɢ����
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
    		//��������
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
    	
    	//��Ϊͬ�෵��ֱ�ӷ���
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
    	
    	//û�п������Ա��Ϊ�����(��ɢ)��ƽ��ֵ(����)
    	if (_isClassification) {
			node.label = most_label(set);
		} else {
			node.label = mean_value(set);
		}
    	if (node.attr_index == null || node.attr_index.length == 0) {
    		return node;
    	}
    	
    	//Ѱ�������и�����
    	node.split_attr = attribute_selection(node);
    	//û�п��Էָ������
    	if (node.split_attr < 0) {
    		return node;
    	}
    	//System.out.println(node.split_attr);
    	int[][] sub_sets = split_with_attribute(node.set, node.split_attr, node);
    	
    	//ȥ����ʹ�õ�����
    	int[] child_attr_index = new int[attr_index.length - 1];
    	int t = 0;
    	for (int index : attr_index) {
    		if (index != node.split_attr) {
    			child_attr_index[t++] = index;
    		}
    	}
    	
    	//�ݹ齨���ӽڵ�
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
    	if (_isClassification) {
    		double reference_value = 0;
        	int max_attribute = -1;
        	for (int attribute : node.attr_index) {
        		double temp = gain_ratio_use_attribute(node.set, attribute);
        		if (temp > reference_value) {
        			reference_value = temp;
        			max_attribute = attribute;
        		}
        	}
        	return max_attribute;
    	} else {
    		double reference_value = -1;
        	int min_attribute = -1;
        	for (int attribute : node.attr_index) {
        		double temp = mse_use_attribute(node.set, attribute);
        		if (reference_value < 0 || temp < reference_value) {
        			reference_value = temp;
        			min_attribute = attribute;
        		}
        	}
        	return min_attribute;
    	}
    	
    }
    
    //��Ϣ���� ID3
    private double information_gain_use_attribute(int[] set, int attribute) {
    	double entropy_before_split = entropy(set);
    	
    	double entropy_after_split = 0;
    	for (int[] sub_set : split_with_attribute(set, attribute, null)) {
    		entropy_after_split += (double)sub_set.length / set.length * entropy(sub_set);
    	}
    	
    	return entropy_before_split - entropy_after_split;
    }
    
    //������ C4.5
    private double gain_ratio_use_attribute(int[] set, int attribute) {
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
    
    private int[][] split_with_attribute(int[] set, int attribute, TreeNode node) {
    	if (_isCategory[attribute]) {
    		//��ɢ����
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
        	
        	//��¼�и��
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
    		//��������
    		double[] points = new double[set.length];
    		for (int i = 0; i < set.length; ++i) {
    			points[i] = _features[set[i]][attribute];
    		}
    		Arrays.sort(points);
    		
    		double entropy = -1;
    		double best_split_point = 0;
    		int[][] result = new int[2][];
    		for (int i = 0; i < points.length - 1; ++i) {
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
