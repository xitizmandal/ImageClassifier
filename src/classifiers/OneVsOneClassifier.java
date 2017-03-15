package classifiers;

import java.io.File;

import org.opencv.ml.CvSVM;

import utils.Utils;

public class OneVsOneClassifier {
	CvSVM classifier;
	
	public OneVsOneClassifier() {
		classifier = new CvSVM();
		classifier.load(new File(Utils.CLASSIFIER_XML_ONEVSONE).getAbsolutePath());
	}

	public CvSVM getClassifier(){
		return classifier;
	}
	
	public void close(){
		classifier.clear();
	}

}
