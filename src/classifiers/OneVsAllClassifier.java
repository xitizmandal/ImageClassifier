package classifiers;

import java.io.File;

import org.opencv.ml.CvSVM;

import utils.Utils;


public class OneVsAllClassifier {
	CvSVM classifier;
		
	public OneVsAllClassifier(int index) {
		classifier = new CvSVM();
		classifier.load(new File(Utils.CLASSIFIER_XML_ONEVSALL + Utils.classification[index] + ".xml").getAbsolutePath());
	}

	public CvSVM getClassifier(){
		return classifier;
	}
	
	public void close(){
		classifier.clear();
	}
}
